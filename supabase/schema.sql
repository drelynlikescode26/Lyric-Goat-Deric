-- Optional hosted persistence for the next phase. The app remains local-first
-- until a Supabase adapter and authentication are configured.
create extension if not exists pgcrypto;

create table if not exists songs (
  id text primary key,
  user_id uuid,
  title text not null default 'Untitled',
  bpm integer not null default 90,
  key text not null default 'auto',
  genre text not null default 'hiphop',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists song_sections (
  id text primary key,
  song_id text not null references songs(id) on delete cascade,
  type text not null default 'verse',
  label text not null default 'Verse',
  lyrics text not null default '',
  rough_text text not null default '',
  phrase_map jsonb not null default '[]'::jsonb,
  versions jsonb not null default '[]'::jsonb,
  settings jsonb not null default '{}'::jsonb,
  audio_path text,
  position integer not null default 0,
  updated_at timestamptz not null default now()
);

create table if not exists writing_feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid,
  song_id text references songs(id) on delete set null,
  section_id text references song_sections(id) on delete set null,
  action text not null check (action in ('accepted', 'edited', 'rejected')),
  source_line text not null default '',
  final_line text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

insert into storage.buckets (id, name, public)
values ('song-audio', 'song-audio', false)
on conflict (id) do nothing;

-- Before a public launch: enable RLS and add user-scoped policies after auth is
-- implemented. Never expose a service-role key in the browser.
