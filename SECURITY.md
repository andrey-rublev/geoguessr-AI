# Security

To report a security problem, use **[Report a vulnerability](https://github.com/andrey-rublev/geoguessr-AI/security/advisories/new)**, which only the maintainer sees, rather than a public issue. You'll get an answer within a few days.

Worth knowing about how it works:

- The app's window talks to a server on your computer that only listens to your computer, and every request must carry a secret the window was opened with. It can start the bot, so a page elsewhere must never be able to reach it.
- The installer and the model come from this repository's releases. The model is checked against its SHA-256 before it's used.
- The bot moves your mouse and presses keys when it plays on your screen. **F8** stops it.
