# Hoop Metrics private passcode gateway
This is a server-side gate for a Vercel Hobby deployment. No provider password-protection add-on is required.

- All routes, including JSON data, CSS/JS, original research page and APIs, pass through `middleware.js`; only the passcode screen, its CSS/JS and `/api/auth` are public.
- PBKDF2-SHA256 verification (150,000 iterations) is performed on the server. The plaintext passcode is **never committed** to the repository or embedded in frontend files.
- Successful entry sets an HMAC-SHA256-signed, HttpOnly, Secure, SameSite=Strict session cookie with a 12-hour expiry. Middleware checks the cookie signature and expiry before allowing any request, including stat feeds.
- Rotate `HM_AUTH_SESSION_SECRET` to invalidate every active session, or `HM_PASSCODE_DERIVED` to change the passcode. Store them only as **encrypted/sensitive** Vercel project environment variables. Values have the format `<16-byte hex salt>:<32-byte pbkdf2 hex hash>` and `<32-byte hex HMAC signing key>` respectively.
- A high-entropy random owner passcode is required. Online attempts are computationally throttled by PBKDF2 but **there is no distributed account lockout** on Hobby. For regulated business use add a global attempt limiter and a managed identity provider.
- Testing must cover both unauthenticated access (302 login for all documents/static bundles, 401 for any API) and authenticated access. Test source URLs too, not just homepage.
- Historic Vercel deployments that **do not include this middleware** would become public if you simply turned off Vercel Authentication on the previous project. For this reason **do not disable SSO on that older project**; launch the passcode-only version as a separate Vercel project and move domains only after verification.
