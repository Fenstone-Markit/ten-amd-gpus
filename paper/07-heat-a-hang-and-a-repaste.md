# Chapter 7: Heat, a hang and a repaste

*Clock caps that are not ceilings, one card's paste, a box that hung after the repaste and came back after a full power drain, a start-up that burned three kilowatts doing nothing, and a system drive that stopped answering for the fourth time. Measured from 2026-10-22 to 2026-10-24 on the ten-card machine, an open-air frame with every card behind one PCIe switch.*

## Findings

1. **The power limit cannot go lower, so the clock has to.** These cards already run at their vendor minimum power limit (294 W or 261 W). Turning on the driver's overdrive features did not lower that floor; it raised the maximum to 402 W. What overdrive does give is a top shader clock you can set, a voltage offset and fan control.
2. **A clock cap is a target, not a ceiling.** With the top clock set to 2,000 MHz, a card under full load ran at 2,161 to 2,248 MHz, 8 to 12 % over the setting. *Added 2026-10-24:* under light load, cards capped at 2,000 or 2,400 MHz reported about 3,050 MHz, above even their stock top clock, so at light load the setting may not be limiting anything.
3. **One card's heat was its paste.** The ASRock card reached 110 C within about 3 seconds of load at 260 to 339 W, with its fans spinning and its clock capped. After a PTM7950 pad it levelled off at 82 to 83 C at 325 W under the same load, with fans around 1,530 rpm.
4. **A phase-change pad wants a gentle first heat.** The test brought it up at 30 % and then 60 % load (57 C and 70 C) before full load, so the pad could melt and settle before it had to carry 325 W.
5. **A card that does not answer is not necessarily a broken card.** After the repaste one card's power controller stopped answering and the whole machine hung. A warm reset did not fix it; a full power drain, every supply off, did. Every card came back.
6. **A missing cache folder cost three kilowatts.** A model start ran all ten cards at about 2,990 W for over two minutes with no progress, reaching 100 C on one card, because a library could not save its tuning results. Giving it a writable folder cut the GPU part of start-up to 48.6 s and the hottest card to 86 C.
7. **The system drive hangs with every power-saving state off.** It stopped answering a fourth time on 2026-10-24, with automatic power-state switching disabled entirely. Its health log is clean. Chapter 4's explanation, a drive that fell asleep and did not wake, is withdrawn.

## Caps

| Setting | Before | After |
| --- | --- | --- |
| Driver feature mask | overdrive off (`0xfff7bfff`) | overdrive on (`0xfff7ffff`), one reboot |
| Power limit range | 261 or 294 W minimum | unchanged minimum, 402 W maximum |
| Top shader clock | 2,835 to 3,035 MHz stock | 2,400 MHz on eight cards, 2,000 MHz on the two hottest |
| Fans at idle | stopped (zero-RPM) | always turning |

The caps are written through the driver's clock table and do not survive a reboot. Under full load the reported clock still ran over the setting, as above. For a hard limit on heat, the levers left are the voltage offset and airflow.

## The ASRock card

The card is G9, the fleet's only ASRock Phantom Gaming RX 7900 XTX, whose poor die contact Chapter 4 already noted. It had always run hot for its power, and during model starts in this period it reached 110 C, the point where the card throttles itself, within seconds. I replaced its paste with a PTM7950 phase-change pad and ran one test on that card alone: five minutes at 30 % busy, five at 60 %, two minutes of rest, then five minutes of a dense bf16 matrix multiply at full load, at the same 2,000 MHz setting as before.

| Under full load at the 2,000 MHz setting | Before the repaste | After |
| --- | --- | --- |
| Peak junction | 110 C within about 3 s | 83 C |
| Power | 260 to 339 W | 325 W, last-minute average |
| Behaviour | still climbing when stopped | flat at 81 to 83 C for 3 minutes |
| Fan | spinning | about 1,530 rpm, with room left |

Same card, same setting, same power, 27 degrees cooler. Selecting one card for a container needs `ROCR_VISIBLE_DEVICES=GPU-<card id>`: passing only the card's render node still showed all ten.

## The hang

After the repaste and power-up, one card logged "SMU: No response" every 5.6 seconds, a second card's fan command timed out, and the machine stopped answering: no SSH, and the fleet console showed it down. All ten cards share one driver, and a card whose power controller stops answering makes the driver wait while holding locks the others need, so everything that touches a GPU, including the console's collector, hung behind it.

I first read that as the card being faulty. It was not shown, and it was wrong: a full power drain, with every supply off, cleared it, and the card has run normally since. A warm reset leaves the cards' power controllers powered, which is the likeliest reason it did not help. That is a reading, not a finding.

## Three kilowatts of nothing

A model start stopped on heat with eight cards at 93 to 100 C. Its log showed weight loading finished, then over two minutes at about 2,990 W on all ten cards without the next step finishing, where the same start in my agent's sandbox takes 28 seconds. The only new message in the log was MIOpen, AMD's convolution library, saying it could not create its database folder (`/root/.config/miopen`). Without a database it cannot save what it measured, and the start never got past that step.

With the database pointed at a writable folder and MIOpen told to pick its algorithms from what it already knows (`MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`, `MIOPEN_FIND_MODE=FAST`), the same start finished its GPU work in 48.6 s with the hottest card at 86 C. My agent's sandbox never showed this because its home folder is writable and holds a database built over weeks.

## Since then (2026-10-24)

- **Clocks at light load.** Cards capped at 2,000 or 2,400 MHz reported about 3,050 MHz while lightly loaded (finding 2).
- **What a power drain resets.** After the full power drain that cleared the hang, the cards' power limits were back at their vendor defaults, and the minimums went back on by hand (2,808 W across the ten cards). Clock caps and fan targets are lost on every reboot.
- **Fan targets.** G7 and the card at bus `e3` had a fan target temperature of 95 C, so their fans stayed slow until the chip was nearly that hot. Both are now 80 C, set by hand and lost on reboot. Through the 2-hour 18-minute GLM soak in [Chapter 8](08-serving-it.md), the hottest reading on any card was 99 C (G7, one 2-second sample), under the 100 C stop.

## The drive, again

[Chapter 4](04-running-it.md) put the system drive's first hang (2026-10-04) down to a power-saving state it could not wake from, and turned the deepest one off. It hung twice more on 2026-10-17, the second time with that state already off, and again on 2026-10-24 with automatic power-state switching off entirely (`nvme_core.default_ps_max_latency_us=0` on the kernel command line).

This time the kernel log reached another machine over the network as it happened, so the sequence is on record:

| Time (MDT) | Kernel log |
| --- | --- |
| 08:19:34 | a write to the drive times out; the kernel aborts it |
| 08:20:04 | the same command times out again; the kernel resets the controller |
| 08:21:25 to 08:22:05 | "Device not ready; aborting reset", three times |
| 08:22:05 | "Disabling device after reset failure"; the root filesystem's journal aborts and it goes read-only |

GLM kept answering from GPU memory for another 75 minutes. Nothing could start on the machine, SSH included, because nothing could be read from the drive. A hard reset brought it back with no loss beyond the writes in flight in those three minutes; the filesystem repaired itself at boot.

| The drive's own health log, after the reset | Value |
| --- | --- |
| Model, controller, firmware | ADATA SX8100NP, Realtek RTS5762, VB411D39 |
| Critical warnings, media errors | 0, 0 |
| Wear | 1 %, 12 TB written |
| Time above its warning temperature, thermal throttling | 0, 0 |
| Error log | says 0 entries; the error-log page itself returns 8 entries of random values |

The board's management controller logged no PCIe error for the drive or its slot at any of the four hangs. A clean health log, a controller that answers "not ready" to every reset, and no link error point at the drive's controller locking up, not at its flash, its heat, its power states or the slot. That is a reading, not a proof: the drive's own error log, which should settle it, is garbage. The drive is being replaced.

## Corrections to Chapter 4

- "A consumer NVMe drive can fall asleep and not wake": withdrawn. The drive also hangs with every power-saving state off. Turning the deepest one off did not fix anything.

## Left open

- **Voltage offsets,** card by card, now that overdrive is on.
- **Caps that survive a reboot,** once the right values are known, with the fan targets and power limits alongside.
- **A new system drive,** with a different controller, then weeks without a hang before calling it fixed.
- **G6,** the card Chapter 4 found 18 degrees hotter than its neighbour at the same power, now that a repaste is known to work here.
