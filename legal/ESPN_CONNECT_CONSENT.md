# In-app consent copy — shown at the "Connect ESPN" step

> This is the single most important legal/UX surface. Show this text at the screen
> where a user pastes their ESPN cookies, with a **required, unchecked-by-default
> checkbox**. Do not let them connect until it's checked. Store a timestamp of the
> consent. Keep the copy in plain English — this is what actually protects you and
> informs the user.

---

### Connect your ESPN account

To pull your league data, [PRODUCT_NAME] needs two values from your ESPN login —
your `espn_s2` and `SWID` cookies. Here's exactly what that means, in plain terms:

**What these are.** They're the session tokens your browser already uses to stay
logged in to ESPN Fantasy. They are **not** your ESPN password, and we never ask
for it.

**What they let us do.** With them, we can read your fantasy leagues — including
private ones — as if we were you. Technically they would also allow changes to
your teams (lineups, adds/drops, trades); **we only ever read. We never make
changes to your ESPN account.**

**How we protect them.** They're **encrypted** before we store them, and are only
ever decrypted in memory to fetch your data. We never show them back to you, log
them, or share them.

**Your control.** You can delete them anytime in Settings, which immediately
removes them from our systems. You can also invalidate them yourself by logging
out of ESPN or changing your ESPN password.

**Important.** [PRODUCT_NAME] is an independent tool. It is **not affiliated with,
authorized, or endorsed by ESPN or The Walt Disney Company.**

- [ ] I understand what my ESPN cookies allow, and I authorize [PRODUCT_NAME] to
      access my ESPN Fantasy data on my behalf. I have read the
      [Privacy Policy](/privacy) and [Terms of Service](/terms).

[ Connect ESPN ]
