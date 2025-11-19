# AI Model Bug Fix Competition - Summary Table

## Quick Reference: All Models Evaluated

| # | Model Name | Branch Status | Commits | Bugs Fixed | Grade | Time | Rank |
|---|------------|---------------|---------|------------|-------|------|------|
| 1 | **CC Opus** (Claude Code Opus) | ✅ Submitted | 12 | 27/60 | C- (45%) | ~60m | 3rd |
| 2 | **CC Sonnet** (Claude Code Sonnet) | ❌ Empty Branch | 0 | 0/60 | F (0%) | N/A | N/A |
| 3 | **CC Sonnet Web** (Claude Code Sonnet Web) | ✅ Submitted | 10 | 49/60 | B+ (81.7%) | ~80m | 🥇 1st |
| 4 | **Gemini CLI** | ❌ No Branch | - | 0/60 | F (0%) | N/A | N/A |
| 5 | **Gemini Jules** (Google Labs Jules) | ✅ Submitted | 1 | 49/60 | B+ (81.7%) | 1h 25m | 🥇 1st |
| 6 | **Gemini Antigravity** | ❌ No Branch | - | 0/60 | F (0%) | N/A | N/A |
| 7 | *Gemini 2.5 Flash* (Unexpected) | ⚠️ Contaminated | 14* | 2/60* | F (3.3%) | N/A | DQ |

*Gemini 2.5 Flash merged Claude Opus work and claimed 58 bugs but only fixed 2 uniquely

---

## Branch Details

| Model | Branch Name | Location |
|-------|-------------|----------|
| CC Opus | `bugfix-claude-opus-20251119` | Local + GitHub |
| CC Sonnet | `bugfix-claude-sonnet-20251119-1000` | Local + GitHub (empty) |
| CC Sonnet Web | `claude/fix-pydantic-v2-config-0118LoWzrgTDnhHBnXUUZLKf` | GitHub |
| Gemini CLI | NO BRANCH SUBMITTED | - |
| Gemini Jules | `bugfix-jules-20251118-2230` | Local + GitHub |
| Gemini Antigravity | NO BRANCH SUBMITTED | - |
| Gemini 2.5 Flash | `bugfix-gemini-2.5-flash-20251119-1200` | Local + GitHub |

---

## Participation Summary

### ✅ Participated (3/6 = 50%)
1. **CC Sonnet Web** - 49 bugs fixed - TIE FOR WINNER
2. **Gemini Jules** - 49 bugs fixed - TIE FOR WINNER
3. **CC Opus** - 27 bugs fixed

### ❌ Did Not Participate (3/6 = 50%)
1. **CC Sonnet** - Created branch but no commits
2. **Gemini CLI** - No branch submitted
3. **Gemini Antigravity** - No branch submitted

### ⚠️ Disqualified (1 unexpected)
1. **Gemini 2.5 Flash** - Contaminated branch (merged others' work)

---

## Key Statistics

- **Expected Models:** 6
- **Actually Participated:** 3 (50%)
- **Winners:** 2 (tied at 81.7% success rate)
- **Total Bugs Available:** 60+
- **Best Performance:** 49/60 bugs (81.7%)
- **Worst Performance (of participants):** 27/60 bugs (45%)
- **No Shows:** 3 models (50%)

---

## Bug Fix Coverage by Winners

| Bug Category | Jules | CC Sonnet Web |
|--------------|-------|---------------|
| Critical (1-20) | 20/20 (100%) | 19/20 (95%) |
| High (21-38) | 18/18 (100%) | 17/18 (94%) |
| Test Fixtures (39-49) | 11/11 (100%) | 8/11 (73%) |
| Medium (50-60) | 0/11 (0%) | 5/11 (45%) |
| **TOTAL** | **49/60** | **49/60** |

---

## Verification Status

| Model | AI Signature Verified | Commit Author |
|-------|----------------------|---------------|
| CC Opus | ✅ Yes | `Co-Authored-By: Claude <noreply@anthropic.com>` |
| CC Sonnet | N/A | No commits |
| CC Sonnet Web | ✅ Yes | Author: Claude |
| Gemini CLI | N/A | No branch |
| Gemini Jules | ✅ Yes | `google-labs-jules[bot]` |
| Gemini Antigravity | N/A | No branch |
| Gemini 2.5 Flash | ❌ No | Jim McMillan (human) |

---

## Final Verdict

### 🏆 Winners (Tied)
1. **Gemini Jules (Google Labs)** - 49/60 bugs, Grade: B+
2. **CC Sonnet Web (Claude)** - 49/60 bugs, Grade: B+

### Recommendations
- **Use Jules branch for production** - cleanest implementation
- **CC Sonnet Web as alternative** - equally effective
- **50% no-show rate** indicates potential scheduling/communication issues

---

*Complete details available in COMPLETE_MODEL_EVALUATION_REPORT.md*