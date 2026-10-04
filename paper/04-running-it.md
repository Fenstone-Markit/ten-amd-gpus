# Chapter 4: Running it

*Notes from operating the ten-card machine and its smaller sibling day to day: a console that showed what spot checks could not, a system drive that went to sleep and did not wake, and a card that runs hot. Measured on 2026-10-04.*

A benchmark tells you what the machine can do for a minute. Running it tells you what goes wrong over weeks. This chapter is the second kind, and most of it is useful on any Linux machine with consumer AMD cards or a consumer NVMe drive.

## Findings

1. **What a monitor reads can be wrong in ways its tests never see.** My agent wrote the first version of a fleet console without being able to see real hardware, so it guessed the kernel's file formats, and its tests passed against its own guesses. On the real machines it found no cards, no temperatures and the wrong link widths. A read-only snapshot of the exact files it reads, from both machines, fixed all three. Two more bugs lived only in the program as it is actually started, not in the library its tests exercised, and the same kind of bug came back once more the same evening. Test the program, not just its parts.
2. **Every RX 7900 XTX reports its link as x16, wherever it sits.** Each card contains its own small PCIe switch, and the GPU reports its link to that switch. The real link is one level further out (below). Read there, all ten cards on the switch run x8 Gen4, as designed, and the refurbished card in the second machine runs x8 in a slot built for x16, probably because of its notched gold fingers. Its earlier load test had read the wrong port and recorded x16.
3. **On ROCm, every process opens every GPU.** A process limited to one card still opens all ten cards' device files and appears in the kernel's per-process records for every card, with zero memory on the nine it does not use. Those records were briefly read as a leak, and written up as the root cause of a failure, before anyone measured a process fenced to one card for comparison. It looked exactly the same. Retracted the same day: judge card use by memory, not by appearance.
4. **A consumer NVMe drive can fall asleep and not wake up.** The system drive's controller stopped answering while its PCIe link stayed up with zero errors. The filesystem shut itself down, nothing on it could be read, not even files still cached in memory, and the kernel's own record of the failure was lost with it; the model kept serving from memory throughout. A cold power cycle through the board's management controller brought it back healthy. Its deepest sleep state is now switched off, at a cost of about 3 W.
5. **One card runs 18 degrees hotter than its neighbour at the same power, with its fan at half speed.** Over long sessions the console's heatmap showed G6 sustaining 94 to 95 °C junction while G5, drawing slightly more, sat at 76 °C. Every card's fan had headroom. G6's power limit can only be lowered by about 10 %, and with it lowered, its first long session stayed below 95 °C with no visible change in the model's speed.
6. **The model's speed has two honest numbers.** The console reports everything the model generated over each few seconds, so it jumps between about 75 and 175 tokens per second as draft acceptance changes with the text. The agent reports one answer over its whole time, including reading a long conversation first, so it reads 100 to 120. Both are right.

## Reading a 7900 XTX's real link

The GPU's own `current_link_width` in sysfs reports 16 on every card, because it describes the link to the card's internal switch. Walk two levels up from the GPU's PCI device to the card's own upstream switch port; that port's `current_link_width` and `current_link_speed` are the link to the slot. On this machine they read 8 and 16.0 GT/s for every card, as the motherboard's switch design intends; on the refurbished card they read 8 in an x16 slot whose other end also supports 16.

## The drive that slept

The system drive is a consumer NVMe drive (an ADATA SX8100NP). It offers five power states:

| State | Power | Into the state | Out of the state |
| --- | --- | --- | --- |
| 0 to 2 | 8, 4 and 3 W | none | none |
| 3 | 0.0128 W | 4 ms | 8 ms |
| 4 | 0.008 W | 8 ms | 30 ms |

Linux's default lets a drive use any state whose combined latency is under 100 ms, so this drive was allowed into state 4, the deepest. Drives in this family are reported to hang when waking from it, which fits what happened: a healthy drive (0 media errors, 100 % spare, 1 % worn) whose controller stopped answering with its link intact.

The fix, live and permanent:

```bash
echo 0 | sudo tee /sys/class/nvme/nvme0/power/pm_qos_latency_tolerance_us
```

and `nvme_core.default_ps_max_latency_us=0` added to the kernel's boot options. One trap: the obvious tool, `nvme set-feature -f 0x0c -v 0`, waits for a 256-byte table on its input when none is given, so in a pasted script it simply hangs. The sysfs route above avoids it.

When it happens, ordinary commands fail with "Input/output error" or "command not found" while anything already in memory keeps running. Do not reboot first: with the shell's own built-ins, `/proc/mounts` shows the root filesystem as `emergency_ro,shutdown` and `/sys/class/nvme/nvme0/state` reads `dead`. Then a cold power cycle, off for 30 seconds; a warm reset may leave the drive hung.

## The hot card

Junction, power and fan, read together during a long session:

| Card | Junction | Power | Fan, against its maximum |
| --- | --- | --- | --- |
| G1 | 80 °C | 305 W | 1,600 of 3,300 rpm |
| G2 | 89 °C | 289 W | 1,495 of 3,500 rpm |
| G3 | 90 °C | 326 W | 1,915 of 3,300 rpm |
| G4 | 84 °C | 289 W | 1,627 of 3,300 rpm |
| G5 | 76 °C | 304 W | 1,369 of 3,300 rpm |
| **G6** | **94 °C** | **290 W** | **1,799 of 3,500 rpm** |
| G8 | 77 °C | 298 W | 1,473 of 3,000 rpm |
| G10 | 76 °C | 305 W | 1,273 of 3,300 rpm |

Cards drawing the same power differ by up to 18 °C, so the difference is not the work: it is air or contact, where each card sits in the frame or how its thermal paste has aged. Another card here, G9, has a diagnosed die-contact fault (it hit 100 °C and kept climbing, with the highest junction and the lowest memory temperature in the fleet) and is kept to short bursts until it is repasted. G6 does not show that signature: its memory runs as warm as its neighbours'. The fan curves are hidden unless the driver's overdrive features are enabled at boot. G6's power limit accepts 261 to 291 W; at 261 W it settles at 260 to 270 W, with short bursts above while the firmware catches up, because the limit governs average power over a window, not every instant. One capped session is not proof; the console will show more.

## What the console shows

| Panel | The question it answers |
| --- | --- |
| Fleet power, energy and efficiency | How hard is the fleet working, and how many tokens does each kilojoule buy? |
| Hottest card and its headroom | Which card is closest to its 110 °C limit, and by how much? |
| Power by machine, over time | When was each machine busy, and how much did it draw? |
| Each model by name | How fast is it now, how much of the window was it working, how many tokens has it served? |
| Card temperatures as a heatmap | Which cards run hot, for how long, and in which sessions? |
| The card table | Each card against its own limits: junction against 110 °C, power against its cap, its real link |

Its first fleet-power figure was wrong by a factor of about three: it added up every reading in each time slot instead of averaging each card first. The test data had one reading every 30 seconds; the real history has one every few seconds. It was caught because a 7 kW peak is impossible on a 20 A, 240 V circuit.

The console also caught the drive failure from the outside: while node02 could barely run a command, its collector, already in memory, kept reporting, and the drive's temperature simply disappeared from the page.

## Left open

- **The fan curves**, and whether the same power cap helps G2 and G3.
- **Why the backup model fails on its two reserved cards.** Each card works alone and the cards can copy to each other; the two-process engine still fails. The first explanation was wrong.
- **The pattern behind three small PCIe incidents** on the network chip's root complex, two of which came on nights when the machine later hung on reboot.
