# Privacy Policy — [PRODUCT_NAME]

> **DRAFT — not legal advice.** Fill the [BRACKETS], and have an attorney review
> before you accept paying customers. Effective date: [DATE].

[PRODUCT_NAME] ("we", "us") is operated by [LEGAL ENTITY / YOUR NAME], [ADDRESS].
This policy explains what we collect, why, how we protect it, and your rights.

## 1. Information we collect
- **Account data:** email address and authentication identifiers (handled by our
  auth provider, [Clerk/Supabase]).
- **ESPN session cookies (`espn_s2`, `SWID`):** you provide these so we can read
  your ESPN Fantasy data on your behalf. See "How we protect ESPN cookies" below.
- **Fantasy data:** league settings, rosters, matchups, and derived stats we fetch
  from ESPN using your cookies, plus recommendations/decisions you make in the app.
- **Usage data:** logs, IP address, and basic analytics needed to run and secure
  the service.
- **Payment data:** if you subscribe, payments are processed by [Stripe]. We do
  not store your card number; we store only a customer/subscription identifier.

## 2. How we use it
To authenticate you, fetch and display your league data, generate stats/answers,
provide the chatbot, process subscriptions, secure the service, and comply with law.

## 3. How we protect ESPN cookies (the sensitive part)
- Your `espn_s2`/`SWID` are **encrypted at rest** using [AES-256 / Fernet]. The
  encryption key is held in a secrets manager, **separate from the database** — a
  database leak alone does not expose your cookies.
- They are decrypted **only in memory**, only to fetch your data, and are **never
  logged, displayed, or shared**.
- We use them **read-only**. We never write to, or make changes in, your ESPN
  account.
- You can delete them at any time in Settings; deletion is immediate and permanent.

## 4. Who we share data with
Service providers who help us operate: hosting/database ([Neon]), authentication
([Clerk/Supabase]), the AI provider that powers the chatbot ([Groq/Anthropic]),
and payments ([Stripe]). We do **not** sell your personal data. We may disclose
information if required by law.

## 5. Data retention & deletion
We keep your data while your account is active. You can delete your ESPN cookies,
individual leagues, or your entire account at any time. Deleting your account
removes your personal data and credentials, except records we must keep for legal
or accounting reasons.

## 6. Your rights
Depending on where you live (e.g., EEA/UK under GDPR, California under CCPA/CPRA),
you may have rights to access, correct, export, or delete your data, and to
withdraw consent. Contact us at [CONTACT_EMAIL] to exercise them.

## 7. Security & breach notification
We use encryption, access controls, and least-privilege practices. No system is
perfectly secure. If a breach affects your data, we will notify you and any
regulators as required by applicable law.

## 8. Children
The service is not directed to anyone under [16/18]; we do not knowingly collect
their data.

## 9. Not affiliated with ESPN
[PRODUCT_NAME] is an independent product and is **not affiliated with, authorized,
sponsored, or endorsed by ESPN, Inc. or The Walt Disney Company.** ESPN is a
trademark of its respective owner.

## 10. Changes & contact
We'll post changes here and update the effective date. Questions: [CONTACT_EMAIL].
