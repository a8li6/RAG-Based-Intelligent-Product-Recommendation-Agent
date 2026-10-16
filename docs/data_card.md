# Data Card: Synthetic Laptop Reference Catalog

| Item | Value |
|---|---|
| File | `data/sample/products.jsonl` |
| Origin | Written by hand for this project. **Fully synthetic and fictional.** |
| Records | 8 (reference fixture for tests) |
| Brand | `DemoLab` (fictional) |
| Created | 2026-10 |
| License / redistribution | Authored for this repository; distributed with it. Not derived from any third-party catalog. |
| Prices | Fixed **snapshot** values in integer USD cents. Not historical or live prices. |
| Stock | Fixed snapshot flag. Not live availability. |
| Source URLs | `https://catalog.example/products/<ID>`. The `.example` domain is reserved; these are sample identifiers, not real retailer links. |
| Timestamp | `updated_at` is a fixed UTC value, so builds are reproducible. |

## Fields

| Field | Type | Notes |
|---|---|---|
| `product_id` | string | `P` followed by 3+ digits; globally unique |
| `title` | string | 1-120 characters |
| `category` | `"laptop"` | the only supported category in version 1 |
| `brand` | string | |
| `price_cents` | integer | >= 0, strict integer; no floats |
| `currency` | `"USD"` | the only supported currency in version 1 |
| `in_stock` | boolean | |
| `ram_gb`, `storage_gb` | integer or null | null means unknown, never zero |
| `weight_kg` | float or null | null means unknown, never zero |
| `gpu_type` | `"integrated"`, `"dedicated"` or null | |
| `use_cases` | list of strings | non-empty |
| `description` | string | up to 2000 characters; restates catalog facts only |
| `source_url` | string | absolute https URL |
| `updated_at` | ISO 8601 datetime | must be timezone-aware |

## Missingness

`P008` has an unknown `weight_kg` by design. A product with an unknown value
cannot satisfy an active constraint on that value, so it is excluded when such a
constraint is active.

## Known limitations and biases

- Tiny and hand-written: suitable for correctness tests, **not** for measuring
  real-world recommendation accuracy.
- Only one category, one brand and one currency.
- Descriptions contain no benchmarks. No performance claims should be derived
  from them.
- Any real catalog added later must document its origin, access date, license
  and redistribution rights here, and keep its license separate from the
  project's code license.
