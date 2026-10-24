# Chapter 6: The memory underneath

*Why GLM-5.3-Flash on ten RX 7900 XTX failed in service: a live buffer gets overwritten after a nearly full card hands memory back. Traced from a crash on the first real prompt, through a GPU code object found inside the model's working memory, to a one-card reproducer with no model at all. Measured from 2026-10-19 to 2026-10-23.*

**Update, 2026-10-24.** GLM now serves on the ten cards: [Chapter 8](08-serving-it.md) is how. The tests in this chapter stand. Three readings are corrected in place below, each marked *Corrected 2026-10-24*: which step triggers the overwrite (finding 5), what ROCm 7.2 shows (finding 6), and whether anything avoids it (finding 7). One of the tests also ran with a mistake of mine in it (in [A marked block](#a-marked-block)).

Chapter 5 ended with GLM answering correctly at a 131,072-token window and one thing unexplained: something was writing garbage into memory it did not own, and building one small tensor on the CPU instead of the GPU had made the symptoms go away. This chapter is what that something turned out to be, as far as I can show it, and why, as of 2026-10-23, it meant the model could not serve on ten cards.

## Findings

1. **Chapter 5's fix hid the fault; it did not remove it.** The first time I served GLM for real, the first prompt of about 2,000 tokens after start-up crashed a worker with an illegal memory access. Every later attempt to serve it failed on its first real requests, in one of three ways: a crash, garbage sequence lengths ("repeats can not be negative"), or a hang.
2. **What gets written is not computed data.** My agent caught the model's sequence-length buffers changing between two points with nothing in between that could write them. The bytes were `.name`, `.size` and MessagePack markers: the first 2 MiB of a GPU code object holding PyTorch's scan kernels, sitting on top of the first 2 MiB piece of PyTorch's memory segment.
3. **Loading those kernels at start-up removed the start-up fault, 8 starts of 8 against 4 of 4 controls faulting.** It did not remove the fault during serving.
4. **The overwrite only ever happens on the cards that are full.** With a marked block placed at the start of PyTorch's small-buffer memory on every card, the block was overwritten on five or six of the six middle-stage cards in every start, during vLLM's memory-profiling step, and never on the two outer stages. The middle stages are the cards vLLM reports at minus 0.03 GiB of headroom; the outer ones have 1.4 and 2.5 GiB.
5. **The trigger is memory being handed back on a nearly full card.** A 40-round test on one card, plain PyTorch, no model: with 150 MiB kept free and memory released between rounds, a block nothing writes to was overwritten from round 8 on, every run. With nothing released, with 4 GiB kept free, or with PyTorch's classic allocator, zero hits. *Corrected 2026-10-24:* unmapping and mapping again is not the trigger. Releasing the memory to the ROCm runtime is: with the released pieces kept inside the process and reused, unmaps and maps still happen and nothing is overwritten. The overwritten page itself is never unmapped or released. See [Since then](#since-then-2026-10-24).
6. **This is a known pattern on RDNA4, not yet fixed.** The same 2 MiB overwrites of live memory are reported for ROCm's virtual-memory calls on gfx1201, and as zeroed live buffers under PyTorch's expandable segments on ROCm 7.14 (19 of 20 runs) but not 7.2 (0 of 30). This chapter adds RDNA3 and a test that runs in seconds. *Corrected 2026-10-24:* ROCm 7.2.4 shows no overwrite on this card either, but not because it is fixed. Memory PyTorch releases there never comes back to the card, so after the first release every later allocation fails and nothing can land on a live buffer. A clean result on 7.2 is not evidence of a fix.
7. **Nothing I could set on ten cards avoids it.** Warm-up order, fresh caches, a lower locked-memory limit, clock caps, a smaller prefill chunk and a smaller cache all only moved where the failure landed. Switching expandable segments off, the one setting that avoids it, runs out of memory on exactly those six cards. *Corrected 2026-10-24:* two things do. Serving GLM as text only gives the middle cards back about 5 GiB, so PyTorch never has to release anything; and on the one-card test, a small library that keeps released memory inside the process removes the overwrite even when PyTorch does release. [Chapter 8](08-serving-it.md)

## The first real prompt

Chapter 5's tests always sent a short warm-up, then a short request, then a long prompt, and they passed. On 2026-10-19 I served the same recipe as a service and sent a short warm-up, a short question, then a prompt of about 1,950 tokens. A worker on the third pipeline stage died with an illegal memory access, and the kernel logged page faults on one card. No test had used that order.

Over the next three days my agent, Itko, ran eight more tasks on it (ports 19 to 26). I kept the rule from Chapter 5: every result is read from its logs before the next brief, and a cause counts only when a run shows it.

| Task | What it showed |
| --- | --- |
| port-20 | At 131,072 in graph mode, 6 of 11 fresh starts faulted, 4 of them before any request; eager mode passed 4 of 4 |
| port-21 | The start-up memory profile only decides where the fault shows: skipping it gave garbage answers from the first request in 4 of 6 servers |
| port-22 | The kernel that crashed read an index of 343,643,167 that the code filling it cannot produce from sane input; a clamp bounded that one read, nothing more |
| port-23 | The sequence lengths were correct after a synchronize and garbage before the next kernel, with no operation in between on that thread; the garbage was the same bytes in every run: `.name`, `.size`, MessagePack string markers |
| port-24 | The garbage was the first 2 MiB of a 3.5 MiB GPU code object (PyTorch's scan kernels), written over the first 2 MiB piece of the segment holding those buffers, identical in 9 of 9 cases. Neither a torch copy nor a torch.distributed call wrote it |
| port-25 | ROCm's API trace, which slows every runtime call, suppressed the fault 6 starts of 6 against 20 of 31 without it: a timing race below PyTorch. The trace also showed PyTorch's memory requests failing and pieces being unmapped and released during start-up |
| port-26 | Calling `torch.cumsum` a few times at worker start, so its code object loads while memory is still free, gave 8 clean starts of 8 against 4 of 4 controls faulting. Eleven servers then answered every test, and a 166-request soak ran with no crash or error until the 3,500 W rule stopped it at 27.7 minutes (three wrong answers on very long needle prompts, not shown to be this fault) |

So the start-up fault was a GPU program being loaded into memory that PyTorch was already using. Why the loader and PyTorch ended up on the same memory was not shown.

## Serving it again

On 2026-10-23 I served the port-26 recipe for my agent to think with. Its first request, a 14,858-token system prompt with tool definitions, crashed a worker two seconds after a long-prompt kernel compiled and loaded for the first time. Each attempt after that changed one thing:

| Attempt | Change | Result |
| --- | --- | --- |
| 1 | the port-26 recipe, one short warm-up | the agent's first request crashed a worker |
| 2 | warm up with long prompts | first 1,600-token prompt: "repeats can not be negative", then a hang |
| 3 | fresh compile caches every start | first 1,600-token prompt: illegal memory access |
| 4 | two tiny prompts first, as my agent's tests did | first answered; the second, 6 s later, crashed |
| 5 | the agent sandbox's 8 MiB locked-memory limit | same as 4 |
| 6 | stock clocks instead of my clock caps | a GPU hardware exception during start-up |

Every change moved the failure; none removed it. The stock-clock run made that plain: slower clocks put it in serving, faster clocks put it back in start-up.

## A marked block

From here the assistant that writes my agent's briefs (Claude) ran the tests on the machine directly, over SSH. Instead of another guess, it had the worker processes plant four 512 KiB blocks with a known pattern right after start-up, before the model or any request data existed. They landed at the start of PyTorch's small-buffer segment, the same place port-24 found overwritten. Each worker then read them, and only read them, before and after every start-up stage.

On every start, five or six of the six workers on the three middle pipeline stages found all 131,072 words of every block changed after vLLM's memory-profiling step: zeros on some, values that read as bf16 numbers between about -4 and 4 on others. The model's own data had landed on memory PyTorch had given to my blocks and never taken back. The four workers on the outer stages never saw a change.

*Added 2026-10-24:* these runs had a mistake of mine in them. The patch that adds the check put it between the two decorators of vLLM's model-step function, so model steps ran without PyTorch's inference mode. That changes how much memory a step holds, not what an overwrite looks like, and the one-card test below, which has no model and none of my patches, shows the same overwrite. Before the mistake was found the middle cards had stopped running out of headroom (Chapter 8), so this exact measurement has not been repeated without it.

vLLM's own log separates the two groups:

| Pipeline stage | Cards | Headroom after profiling (vLLM's estimate) | Blocks overwritten |
| --- | --- | --- | --- |
| PP0 | 2 | +2.48 GiB | never |
| PP1, PP2, PP3 | 6 | -0.03 to +0.03 GiB | every start |
| PP4 | 2 | +1.36 GiB | never |

With expandable segments off, the same six cards ran out of memory outright: 578 MiB asked for, 434 MiB free.

## One card, no model

If full cards and memory being handed back are the trigger, it should show without GLM. The test fills one card with "weights", plants the same four blocks, then in each round allocates activation-sized buffers full of random bf16 values until a set amount of memory is left free, checks the blocks, and frees the round's buffers:

| Allocator and release | Free memory kept | Blocks overwritten in 40 rounds |
| --- | --- | --- |
| expandable segments, memory released every other round | 150 MiB | 12 and 48 in two runs; first at round 8 every time |
| expandable segments, memory released every other round | 4,096 MiB | 0 (20 rounds) |
| expandable segments, nothing released | 150 MiB | 0 |
| expandable segments, no explicit release, but allocating until PyTorch runs out | 150 MiB | 4, at round 0 |
| PyTorch's classic allocator | 150 MiB | 0 (in 20 of the 40 rounds it ran out of memory before reaching the 150 MiB mark, so its pressure was a little lower) |
| `HSA_ENABLE_SDMA=0` or `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` | 150 MiB | no change |

PyTorch hands pages back in two ways: when told to (`empty_cache()`), and by itself when an allocation fails. The fourth row is the second way, and it is why making `empty_cache()` do nothing did not help GLM: on a full card PyTorch releases pages on its own.

The test, its runner and its full results are in [tools/vmm-repro/](../tools/vmm-repro/). It runs in seconds, so a fix can now be tested in a minute instead of a seven-minute model start.

## What is shown, and what is not

Shown: on these cards, with this ROCm (10.0, TheRock build, HIP 7.15) and PyTorch 2.12, releasing and remapping expandable-segment pages on a card within about 150 MiB of full overwrites live memory in whole 2 MiB pages. GLM's three middle pipeline stages run at that edge on ten cards, which accounts for every failure in this chapter and the last: the code object in port-24 is just whatever got the pages that time.

Not shown: which layer reuses the pages under a live mapping. The RDNA4 report suspects the GPU keeps a stale address translation after an unmap. A reading of the ROCm runtime source pointed at the runtime freeing the driver allocation behind a mapped handle. Neither is proven here. *Corrected 2026-10-24:* narrowed, still not named. See the next section.

Related reports:

- [ROCm/legacy-rocm-build#6692](https://github.com/ROCm/legacy-rocm-build/issues/6692): on gfx1201, reusing a virtual address after `hipMemUnmap` overwrites live buffers in whole 2 MiB pages; also when VRAM fills up with nothing unmapped.
- [ROCm/TheRock#9274](https://github.com/ROCm/TheRock/issues/9274): a live buffer reads back as zeros under PyTorch's expandable segments on ROCm 7.14, not on 7.2; one user reports a 7900 XT.
- [pytorch/pytorch#196258](https://github.com/pytorch/pytorch/issues/196258): a separate expandable-segments bug, where `empty_cache()` can unmap memory with GPU work still queued on it.

## Since then (2026-10-24)

Six more tests on the same card, all in [tools/vmm-repro/](../tools/vmm-repro/), narrowed it down without naming the layer:

| Test | Result |
| --- | --- |
| A logging library in front of PyTorch's four virtual-memory calls, watching the page that holds three of the four blocks | That page's handle is created and mapped once, then never unmapped, never released and never mapped over. Each of its three blocks was overwritten 16 times in 40 rounds |
| The same page mapped a second time, at a second address, and read through both | Every hit showed through both addresses, identically. The memory behind the live handle changes, not just PyTorch's view of it |
| After each release, a plain `hipMalloc` (not PyTorch) filled with `0xAB` | The live blocks read back `abababab`: an unrelated allocation was given memory a live tensor still owns, with 798 MiB of the card free |
| Released handles kept in the process and handed back to PyTorch's next request of the same size, so no release reaches the runtime | 0 hits in two runs, no out-of-memory, 13 buffers every round; unmaps and maps still happen |
| Released handles dropped instead of released (a leak) | 0 hits, but out of memory from round 2, the same pattern as ROCm 7.2.4 |
| The same sequence of calls in plain HIP, including an exact replay of an 11,314-call PyTorch trace at the same addresses | 0 hits. Something else PyTorch does is part of the trigger |

So the overwrite needs a release to reach the runtime, and what gets overwritten is memory behind a handle that is still live and still mapped. Whether the runtime frees the wrong physical memory, or the kernel driver hands out pages that still back a live handle, is not shown. PyTorch does not overwrite it directly: it never unmaps, releases or maps over that page. Something it does is still part of the trigger, since plain HIP making the same calls does not set it off.

## Corrections to Chapter 5

- "A 321B model serves 131,072 tokens on ten consumer cards" held for the tests in that chapter, not for service. GLM answered correctly in every test run, and as of 2026-10-23 it did not survive its first real requests after a fresh start ([Chapter 8](08-serving-it.md) is how that changed).
- "One speed shortcut broke every window above 32,768": building the chunk counts on the CPU removed that symptom, but the counts were being overwritten from outside, by the fault in this chapter. Chapter 5's finding 4 already said the writer was not named; it is now known in kind, if not in component.

## Left open

- **A driver-side fix.** The amdgpu driver has settings for how GPU page tables are updated. Each needs a reboot, then one run of the one-card test answers whether it helps. The layer at fault is still not named.
- **Twelve cards.** With the eleventh and twelfth cards, GLM runs as six pipeline stages of seven or eight layers, and every card has room to spare. That avoids the trigger rather than fixing it. *2026-10-24:* text-only serving already gives the middle cards room on ten (Chapter 8).
- **The upstream report,** with the one-card test, filed against ROCm.
- **The decision on Chapter 5's patches.** Once a fix holds on the one-card test, the switches that only moved this fault around (the CPU-built chunk counts, the start-up prewarm, the forced profile) should come out one at a time, keeping only what is still needed.
