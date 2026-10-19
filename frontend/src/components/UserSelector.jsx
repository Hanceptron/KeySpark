// Native select styled to match the dark dashboard. Three states:
// - users === null (pre-first-fetch): render nothing so the header
//   does not show a placeholder before the API has returned.
// - users === []  (fetched but empty): render disabled with a
//   "no users" label so the slot still exists and the surrounding
//   layout does not jump when the first user appears.
// - users non-empty: an active dropdown.
export function UserSelector({ users, value, onChange }) {
  if (users === null) return null;

  if (users.length === 0) {
    return (
      <select
        disabled
        aria-label="Select user (none available)"
        className="bg-zinc-800/60 border border-white/10 text-zinc-500 text-sm rounded px-2 py-1"
      >
        <option>no users</option>
      </select>
    );
  }

  return (
    <select
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value)}
      aria-label="Select user"
      className="bg-zinc-800/60 border border-white/10 text-zinc-200 text-sm rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-brand-cyan/70"
    >
      {users.map((u) => (
        <option key={u} value={u}>
          {u}
        </option>
      ))}
    </select>
  );
}
