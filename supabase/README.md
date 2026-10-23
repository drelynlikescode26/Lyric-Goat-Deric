# Optional Supabase phase

The current application stays local-first and requires no Supabase spending.
`schema.sql` prepares songs, sections, private audio storage, and writing
feedback for a later migration.

Do not connect this for a public release until authentication and user-scoped
row-level-security policies exist. The service-role key must stay server-side.
