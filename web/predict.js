// Prediction: what you're likely to say next, and completions while typing.
// Pure functions over a small "habits" object; app.js stores it.

const RECENT_MS = 10 * 60 * 1000;   // a phrase this recent counts as "just said"

export function emptyHabits() {
  return { total: {}, hours: {}, after: {}, last: null, lastAt: 0 };
}

export function recordUse(h, phrase, when = new Date()) {
  h.total[phrase] = (h.total[phrase] || 0) + 1;
  const hrs = h.hours[phrase] || (h.hours[phrase] = new Array(24).fill(0));
  hrs[when.getHours()] += 1;
  if (h.last && h.last !== phrase && when.getTime() - h.lastAt < RECENT_MS) {
    const row = h.after[h.last] || (h.after[h.last] = {});
    row[phrase] = (row[phrase] || 0) + 1;
  }
  h.last = phrase;
  h.lastAt = when.getTime();
  return h;
}

export function forget(h, phrase) {
  delete h.total[phrase];
  delete h.hours[phrase];
  delete h.after[phrase];
  for (const row of Object.values(h.after)) delete row[phrase];
  if (h.last === phrase) h.last = null;
  return h;
}

// 0..1 score per phrase: what usually follows the last thing you said,
// what you say at this time of day, and what you say most overall.
export function scores(h, phrases, when = new Date()) {
  const out = {};
  if (!phrases.length) return out;
  const sum = (o) => Object.values(o).reduce((a, b) => a + b, 0);

  const totalAll = sum(h.total);
  if (!totalAll) return out;

  const hour = when.getHours();
  const near = (p) => {
    const hrs = h.hours[p];
    if (!hrs) return 0;
    return hrs[(hour + 23) % 24] + hrs[hour] + hrs[(hour + 1) % 24];
  };
  const hourAll = phrases.reduce((a, p) => a + near(p), 0);

  const recent = h.last && when.getTime() - h.lastAt < RECENT_MS ? h.after[h.last] : null;
  const recentAll = recent ? sum(recent) : 0;

  for (const p of phrases) {
    const overall = (h.total[p] || 0) / totalAll;
    const timeOfDay = hourAll ? near(p) / hourAll : 0;
    if (recentAll) {
      const next = (recent[p] || 0) / recentAll;
      out[p] = 0.5 * next + 0.3 * timeOfDay + 0.2 * overall;
    } else {
      out[p] = 0.6 * timeOfDay + 0.4 * overall;
    }
  }
  return out;
}

// Up to n phrases, but only ones at least `minShare` as likely as the top
// one, so a weak runner-up doesn't just repeat the full list.
export function likelyNext(h, phrases, when = new Date(), n = 4, minShare = 0.35) {
  const s = Object.entries(scores(h, phrases, when)).sort((a, b) => b[1] - a[1]);
  if (!s.length || s[0][1] <= 0) return [];
  const bar = s[0][1] * minShare;
  return s.filter(([, v]) => v > 0 && v >= bar).slice(0, n).map(([p]) => p);
}

// Everyday phrases offered while typing, on top of your own.
export const COMMON = [
  "Yes", "No", "Maybe", "Okay", "Thank you", "Thanks a lot", "Please",
  "Sorry", "Excuse me", "Hello", "Hi", "Goodbye", "See you later",
  "Good morning", "Good night", "How are you", "I'm fine", "I'm good",
  "I'm tired", "I'm hungry", "I'm thirsty", "I'm cold", "I'm hot",
  "I'm in pain", "I feel sick", "I need help", "I need the toilet",
  "I need a break", "I need water", "I need my phone", "I don't know",
  "I don't understand", "I understand", "I agree", "I disagree",
  "I love you", "I'm on my way", "I'll be there soon", "I'm busy",
  "Not right now", "Wait a moment", "Hold on", "Just a minute",
  "Can you repeat that", "Can you help me", "Can you call me",
  "Can you speak slower", "Can you write it down", "Please stop",
  "Stop", "Go", "Come here", "Leave me alone", "Let's go",
  "What time is it", "What's the weather", "What day is it",
  "What's for dinner", "Where are you", "Where is it", "When",
  "Who is it", "Why", "How much is it", "Tell me a joke",
  "Turn it up", "Turn it down", "Turn off the light", "Open the door",
  "Call an ambulance", "Call my family", "Text me", "Read it to me",
  "Play some music", "Set a timer", "Remind me later",
];

// Completions for what's been typed so far. Your own phrases come first,
// then the most-used, then shortest.
export function completions(text, own, extra, usage = {}, n = 6) {
  const q = text.trim().toLowerCase();
  if (!q) return [];
  const words = q.split(/\s+/);
  const seen = new Set();
  const pool = [];
  for (const p of own) pool.push({ p, mine: 1 });
  for (const p of extra) pool.push({ p, mine: 0 });

  const out = [];
  for (const { p, mine } of pool) {
    const key = p.toLowerCase();
    if (seen.has(key) || key === q) continue;
    seen.add(key);
    let score = 0;
    if (key.startsWith(q)) score = 3;
    else if (wordsInOrder(key.split(/\s+/), words)) score = 2;
    else if (key.includes(q)) score = 1;
    if (score) out.push({ p, score, mine, used: usage[p] || 0 });
  }
  out.sort((a, b) => b.score - a.score || b.mine - a.mine
    || b.used - a.used || a.p.length - b.p.length);
  return out.slice(0, n).map((x) => x.p);
}

// "i nee he" matches "I need help": every typed word starts a phrase word, in order.
function wordsInOrder(phraseWords, typed) {
  let i = 0;
  for (const w of phraseWords) {
    if (i < typed.length && w.startsWith(typed[i])) i++;
  }
  return i === typed.length;
}
