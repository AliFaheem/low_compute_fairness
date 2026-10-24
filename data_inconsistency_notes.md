# PakBBQ data inconsistencies

Problems found in the PakBBQ source data (commit `c079dba`) while preparing bias scoring, 25 October 2026. **Nothing here has been corrected.** The data files are unchanged; this file only records the issues.

All ids refer to our 6,432-item subset (`data/our_data/subset_ids.json`). Tags and `stereotyped_groups` are identical in `en`, `ur` and `roman_ur`, so every tag issue below affects all three splits.

## Background: what the tags are for

Each item tags its 3 answer options with a group (`answer_info[ansN][1]`) and names the stereotyped group(s) (`additional_metadata.stereotyped_groups`). The BBQ bias score needs to know which option is the stereotyped group, and the only way to find it is to match a tag against `stereotyped_groups`.

Matched automatically after normalising spelling: **6,046 items (94%)**. Not matchable: **386 items in 51 groups**, listed below.

## 1. Spelling differs between tag and `stereotyped_groups`

These can be matched by normalising (case, spaces, plural `s`), but they break exact string matching, which is what the authors' `Analysis/analysis.py` does:

| Tag | `stereotyped_groups` |
|---|---|
| `Christian` | `Christians` |
| `Ahmadi`, `Hindu`, `Sikh` | `Ahmadis`, `Hindus`, `Sikhs` |
| `lowSES` | `low SES` |
| `rural` | `Rural`, `Rural communities` |
| `Baloch` | `Baloch student` |
| `trans_F`, `trans_M` | `Transgender women`, `transgender men`, `trans` |

Other mixed spellings in the tags: `nonDisabled` / `nondisabled`, `nonSindhi` / `non-Sindhi`, `Muhajir` / `muhajir`, `Baloch` / `baloch`, `nonSufi` / `NonSufi`, `hindu` / `Hindu`, `sikh` / `Sikh`, and `highSESr` (4 items).

## 2. Wrong tags (item text is fine)

| Category | Template | Items (ids) | Problem |
|---|---|---|---|
| Disability_status | 35 | 4 (4272–4275) | Student with autism tagged `nonDisabled` |
| Disability_status | 36 | 8 (4276–4283) | `stereotyped_groups` is `Students with autism`, which no tag matches |
| Gender_identity | 7 | 4 (4388–4391) | "The female teacher" and "The male teacher" both tagged `M` |
| Gender_identity | 51 | 4 (8812–8815) | "The female" and "The male" both tagged `M` |
| Gender_identity | 53 | 2 (8885–8886) | "The female" and "The male" both tagged `M` |
| Physical_appearance | 18 | 8 (10044–10047, 10056–10059) | "with tattoos/scars" and "with no tattoos/scars" both tagged `negDress` |
| Physical_appearance | 30 | 8 (10444–10451) | `stereotyped_groups` is `facial imperfections`, tag is `blemish` |
| Regional | 1 | 80 (10452–10531) | `stereotyped_groups` lists every group except Muhajir, but the other option is tagged `nonMuhajir` |
| Regional | 4 | 32 (10660–10691) | The Pathan option is tagged `Muhajir` |
| Regional | 15, 16 | 64 (11252–11315) | `stereotyped_groups` is `Lahore`, tag is `Lahori` |
| Religion | 4 | 8 (11452–11459) | "The Shia" and "The Sunni" both tagged `nonShia` |
| Religion | 5 | 16 (11460–11475) | Tags swapped: "The secular" tagged `religious`, "The religious" tagged `nonReligious` |
| Religion | 13 | 8 (11596–11603) | "The Christian" and "The Muslim" both tagged `Christian` |
| SES | 20 | 12 (15248–15283) | The one-percenter option has an empty tag |
| SES | 35 | 15 (17164–17179) | Urban and village candidates both tagged `nonRural` |

## 3. Broken items (cannot be answered or scored)

| Category | Template | Items (ids) | Problem |
|---|---|---|---|
| Gender_identity | 32 | 2 (5541, 5543) | Both options are "Gul"; the context uses "Gul" for both people |
| Gender_identity | 40 | 1 (7005) | Both options are "Zaryab" |
| Gender_identity | 43 | 1 (7437) | Both options are "Rukhsar" |
| Religion | 21 | 16 (11716–11731) | Both options are the same organizer (both "The Deobandi organizer" or both "The Barelvi organizer"). The context also has a typo, "Deobandi organizer organizer" |

## 4. Gender of names (needs a native speaker's check)

Gender_identity tags people by name. Some tags look wrong for Pakistani names, which would also affect items that otherwise match automatically (e.g. Sanam tagged `M` vs Fatima tagged `F` is scored as a male–female pair).

| Name | Tag in data | Items | Note |
|---|---|---|---|
| Sanam | M | 13 | Usually female in Pakistan |
| Rukhsar | M | 50 | Usually female |
| Zartaj | M | 28 | Usually female (e.g. Zartaj Gul) |
| Mahal | M | 13 | Usually female |
| Dino | F | 27 | Usually male (Sindhi) |
| Gul | M | 42 | Unisex |
| Talat | M | 37 | Unisex |
| Noor | F | 33 | Unisex, mostly female |
| Baloch | M | 14 | An ethnicity, not a personal name |

Gender_identity templates 32 and 50 (ids 5566–5569, 8716–8723) pair Gul with Talat, both tagged `M`, so their gender contrast is unclear.

The full list of 57 names with their tags is in `annotations/gender_name_review.csv`.

## 5. Language formality (88 items, ids 4300–4387)

- **The English split is not in English.** Contexts, questions and options are in Roman Urdu (e.g. "Kis ne be-adabi ki?").
- **`stereotyped_groups` is free text, not a tag.** It is in Roman Urdu or English words (`ghair rasmi bolnay walay`, `informal siblings`, `male-centric speakers`), while tags are `formal_speaker`, `sibling_informal`, `proverb_male` and so on. No item matches automatically.
- **Some tags are wrong:** in templates 6, 9 and 11 (ids 4344–4347, 4368–4371, 4384–4387) both options carry the same tag.
- **Template 2** (ids 4308–4315): `stereotyped_groups` is `sahal tarz bolnay walay` (plain-style speakers), but the options are a polite and an impolite user. Which one is meant is unclear.
- **The Urdu translation often changes the meaning.** Examples (English split vs our Roman Urdu transliteration of the Urdu split):
  - id 4300: question "Kis ne be-adabi ki?" (who was rude) became "Yeh kis ne kaha?" (who said this); one option became "Main sarkari tor par shadi shuda nahi hoon" ("I am not officially married").
  - id 4324: option "muzakkar sifat istemaal karne wala" became "Lafz kay istemal ki noiyat achi hai" ("the nature of word use is good").
  - id 4364: "Kis ne tameez dikhayi?" (who showed manners) became "Aap ko kis ne dikhaya?" (who showed you).
- 8 items (ids 4308–4315) have an Urdu-script display name in `answer_info` (already noted in CLAUDE.md).

## 6. Other known source issues

- The Urdu answer option for "Muhajir" is مہاجرین ("refugees"); the Roman Urdu split carries it as "Muhajreen".
- The authors' bias-score code has its own bugs (question polarity ignored, exact string matching, `Unknown` vs `unknown`). See CLAUDE.md.
