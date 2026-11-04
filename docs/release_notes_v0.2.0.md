# Release notes: v0.2.0 (Persian query prototype)

Adds Persian (Farsi) **query** understanding on top of v0.1.0. Responses stay English and the catalog is
not translated.

## Added

- Persian normalization (Persian and Arabic digits, `٫` `٬` `،`, slash decimals, Arabic letter variants).
- Persian constraint phrases for budget, RAM, weight and stock, with the same clarification questions as English.
- Currency policy: only USD; amounts in toman, rial, euro or pound are asked about, never ignored or guessed.
- A Persian concept lexicon and a translation table to the catalog's own use-case vocabulary.
- A 60-query Persian evaluation set parallel to the English one, raw reports, a generated analysis
  (`docs/evaluation_fa.md`) and a native-speaker review sheet (`docs/fa_review_sheet.md`).

## Measured (held-out Persian test split, one run)

Before any support, 22 of 120 returned products broke a constraint the user had stated; after, 0 of 96, with
all 33 constraint readings correct. Retrieval Recall@10 rose from 0.455 to 0.739. English results are
unchanged (a test compares a fresh run with the published report).

## Not verified / known limits

- The Persian query texts were written by an AI assistant and **have not been reviewed by a native speaker**;
  the lexicon shares their author, so results are optimistic.
- A Persian request that uses no word in the lexicon retrieves nothing meaningful; negation and abstention
  on irrelevant queries do not work (as in English).
- Not covered: transliterated Persian in Latin letters, colloquial spellings, number words, Persian responses.
- Persian requests take about 1.4x the time of their English twins (about 1.5 ms against 1.2 ms in-process).
