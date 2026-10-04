# Chapter 3: A tutor on one refurbished card

*A second machine, built for a different kind of work. Frankenstein: a Threadripper 3960X with one refurbished RX 7900 XTX, running `cyankiwi/Qwen3.5-9B-AWQ-4bit` on the same image as the brain (`localhost/n02-brain:rc3c`), with a 65,536-token window and three draft tokens. The ten-card machine from the earlier chapters, and its Qwen3.8-27B, helps when a question needs it. Measured on 2026-10-03 and 2026-10-04.*

Everything before this chapter was about one agent's speed. This one is about a tool for people: a homework tutor for a few teenagers in my family, reached over WhatsApp, running on hardware I own, under rules their parents read before it was switched on.

I wanted it to be honest more than I wanted it to be clever. A tutor that confidently tells a fourteen-year-old something false is worse than no tutor. Most of this chapter is about that problem, because it turned out to be the hard one.

## Findings

1. **One 24 GB card runs a useful tutor, but not the one I wanted.** The 9B model fits with its full window and reads photos. Qwen3.6-35B-A3B, which knows far more and is fast, does not fit: on one card it ran out of memory while starting, with 23.57 of 23.98 GB in use, even with its window cut to 16,000 tokens. It needs two cards, which is where the next card goes.
2. **A prompt can fix how a small model talks, but not what it knows.** Five prompt versions fixed the tone, the length, the questions it asked and when it looked things up. None of them stopped it inventing facts it did not have: asked how The Last of Us Part II ends, it invented scenes twice, the second time under an instruction to say when it did not know.
3. **Search alone did not fix it either.** Five search results gave the model about 1,450 words from proper gaming outlets, and both the 9B and the 27B still invented the details the snippets left out. Snippets are written to sell a click, not to tell a plot. Strict and moderate safe search returned almost the same sites, so the filter was never the limit.
4. **What fixed it was splitting the work.** Given a full Wikipedia plot section, the 27B answered almost exactly right; the 9B, given the same text, asked to search again. So the 9B now decides when a question needs looking up (3 of 3 lookup questions and 0 of 4 ordinary ones in testing), the app fetches the material, and the 27B on the ten-card machine writes the answer. Asked who won the 2026 Stanley Cup, the tutor answered with the team, the record, the playoff MVP and the coach's place in history, all correct against the league's own report.
5. **No source is the truth by name.** One proposal in review was to tell the model to trust Wikipedia when sources disagreed. I turned it down: Wikipedia is a quick, often shallow summary, and trusting it by name would only swap one bias for another. The rules now separate what happened from what people think it means, state a fact only when two sources agree or an official one says it, and say so when sources disagree. Every source is labelled by kind: official, encyclopedia, news, community.
6. **The channel decided more than the model did.** Meta's terms have barred general-purpose AI assistants from WhatsApp's Business API since 15 January 2026, and my test account was locked before I learned that. WhatsApp's newer personal Agent Platform works: each kid creates an agent on their own phone, only its creator can talk to it, and the tutor polls for messages without opening any port to the internet. The tutor receives each message through Meta's API, so the conversation is not end-to-end encrypted between the child and the tutor the way a chat between two phones is.
7. **The model's sampling was never set.** The app sent no temperature or other values, so the 9B chatted with the settings meant for its long-reasoning mode. Its model card gives a chat profile (temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5), and with it replies came out shorter and steadier, with no drift into other languages.
8. **Every audit found something real.** My agent writes the code and I audit it before it reaches a child. The audits found the WhatsApp key sent to whatever server a redirect pointed at (Python's `urllib` keeps the Authorization header across hosts), a message that could vanish if one check failed at the wrong moment, the search path skipping the safety check, replies that gave up with "I couldn't look that up", postal codes and spaced phone numbers reaching the search engine, and a raw internal search command sent to a student when search was switched off. Each fix carries a test that fails on the old code.

## How I tested

Prompts were tested against the model's API alone, with eight fixed questions (an introduction, homework, a game question, a sensitive question and so on), each in a fresh conversation, so one bad answer could not poison the next. Search and the answer-writing step were tested the same way, with the same material handed to both models. Then everything was tested live, from a spare phone, through WhatsApp, as a kid would use it. A long conversation full of earlier invented answers kept the model inventing; "new topic", which clears the conversation, fixed that, so every test started clean.

## What each prompt version fixed

| Version | What changed | What it fixed |
| --- | --- | --- |
| v3 | the first full rule set, with a fixed crisis section | safety, but it called itself "a dedicated tutoring engine" and recited its own rules |
| v4 | a persona and a tone | it recited the persona text back instead |
| v4c | one fixed opening line, one question per reply, about 120 words | the tone the kids actually wanted |
| v4d | how to look things up, near the top, with three worked examples | lookups went from 0 of 3 to 3 of 3, with no false searches |
| v4e | search results come from the app, never from the student, and must not be extended by guesses | fewer gaps filled with invention |

## The same question, four ways

"What happens at the end of The Last of Us Part II?"

| Model | Material | Result |
| --- | --- | --- |
| 9B | none | invented scenes and characters |
| 9B and 27B | five search snippets, about 1,450 words | both invented the missing pieces |
| 9B | Wikipedia's plot section | asked to search again |
| 27B | Wikipedia's plot section | almost exactly right |
| Live, after the split | snippets, the plot section and the 27B writing | the main events right; one flashback misattributed and one fan interpretation stated as fact, which the source rules in finding 5 were written to stop |

## How a question flows

| Step | Who | What |
| --- | --- | --- |
| The question arrives | WhatsApp's Agent Platform | polled by the tutor app; nothing listens on the internet |
| Does it need facts? | the 9B, on the refurbished card | answers ordinary homework itself; writes one search request otherwise |
| Look it up | the app | removes names, phone numbers, emails, addresses and postal codes; searches with strict safe search; fetches the relevant Wikipedia section when there is one; labels every source |
| Write the answer | the 27B on the ten-card machine | under the source rules; if it does not answer within 30 seconds, the 9B writes it instead |
| Before sending | the app | the output check, a list of sources, WhatsApp formatting |
| If anything fails | the app | the 9B still answers, from what it knows, and says it could not double-check |

## The rules the parents read

The tutor talks openly about what teenagers actually care about: the games they play, including the violent ones, real history including wars, horror, how the body works, and what drugs and vaping do. It has four hard lines whatever the framing: no sexual content, no real-world instructions that could hurt someone, nothing about self-harm methods, and no asking for or sharing personal details. If a child seems upset or unsafe it stops being a tutor: it answers warmly, asks whether they are safe, points them to a trusted adult and to Kids Help Phone, and never argues with the feeling. The parents received the full instructions, word for word.

## What it costs

| Item | Measured or quoted |
| --- | --- |
| The card | idles at about 10 W; its draw while answering is not measured yet |
| Search | Brave's API, about $5 per thousand searches; each kid is capped at 30 a day |
| A looked-up answer | a few seconds longer than an ordinary one, because the 27B writes it |
| The counts | lookups, searches, answers written by the 27B, fallbacks and failures, kept per day with no message content, shown on the fleet console |

## What I got wrong

- **I tried the bigger model first.** The 35B looked like the answer to the knowledge problem, and it does not fit on one card. Measuring before planning would have saved an evening.
- **I built on the wrong channel.** I did not read the Business API's terms before building on it. The lesson is in my working rules now: check a platform's terms before writing code for it.
- **The first build would have leaked a key.** The redirect bug in finding 8 was in the code my agent wrote and passed its own tests. It was caught by an audit, not by luck, which is why every version is audited.

## Left open

- **A real safety check on every answer.** The check runs on every path, but it is still a placeholder that lets everything through; today the instructions are the only safeguard, and the parents know the instructions word for word.
- **Photos of real homework.** Both models read a printed worksheet correctly. Handwriting, angles and shadows are untested.
- **Documents.** A syllabus arrives as a PDF; reading it is app work, not model work, and not built yet.
- **The 35B on two cards**, when the next card arrives: more knowledge, documents and vision in one model on Frankenstein, and the ten-card machine left to its agent.
- **The tutor's code** is not in this repository yet.
