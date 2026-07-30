# FMA V7.1 Native Paper Delivery

## Outcome

V7.1 defines a small post-S6 publication layer: a fresh Codex author is
requested as `gpt-5.6-sol`, while the FMA harness owns the evidence envelope,
typed value substitution, manifest validation, build, page rendering, audit
receipts, and release postcondition.

The first-principles choice is deliberately simple. The design uses a frontier
model's general writing and LaTeX capability instead of adding a large bespoke
prose engine. Reliability comes from binding its paper to authenticated
workflow evidence and repeatedly observing the actual PDF.

The V7.1 publication path is defined by:

- `fma/v7_1/paper_schemas.py`;
- `fma/v7_1/paper_runtime.py`;
- `fma/v7_1/paper_role_driver.py`;
- `fma/v7_1/paper_renderer.py`;
- `fma/v7_1/paper_cli.py`;
- `fma/v7_1/templates/fma_article_v1.tex`;
- `fma/v5/task_template/skills/paper-authoring-native/SKILL.md`;
- this delivery contract.

## Absorbed design from the reference harness

The reference
[`Ephemeral6/modeling-harness` at `e6b4e53`](https://github.com/Ephemeral6/modeling-harness/tree/e6b4e53e5210f5a231993f57e72dd8c36217570e)
contributes three strong ideas:

1. a paper follows the reader's questions rather than the chronological work
   log, encoded in
   [`templates/config/narrative.json`](https://github.com/Ephemeral6/modeling-harness/blob/e6b4e53e5210f5a231993f57e72dd8c36217570e/templates/config/narrative.json);
2. the writer receives a narrative brief built only from admitted evidence;
3. a
   [separate paper verifier](https://github.com/Ephemeral6/modeling-harness/blob/e6b4e53e5210f5a231993f57e72dd8c36217570e/templates/prompts/roles/paper-verifier.md)
   checks every consequential statement after authoring.

FMA retains those ideas and strengthens their authority boundary. An
authenticated workflow-evidence node is still not permission for the writer to invent a number,
recreate a plot by hand, resolve a citation from memory, or promote the claim
ceiling. Those effects remain code-owned and content-addressed.

Claim-ledger `statement` and `scope_qualifier` fields are deliberately
number-free, TeX-free semantic summaries. Exact values bind through
`numeric_token_ids` and appear only in manuscript prose via
`\FMAValue{...}`. This keeps manifest repair instructions unambiguous.

## Six-question narrative

| Reader question | Required paper function | Evidence obligation |
|---|---|---|
| What must actually be decided? | Reframe the prompt as a falsifiable decision problem and success contract. | Frozen S0 problem, decision object, metric, and failure exit. |
| Where is the information difficulty? | Explain the structural obstacle rather than merely saying the task is complex. | Data limitations, identifiability, coupling, stationarity, or computational evidence. |
| How does the model address it? | Present intuition before variables, equations, assumptions, and solver details. | Model specification, dependency graph, executable artifact, and necessity of each component. |
| Why should the result be trusted? | Form a credibility chain instead of listing plots. | Invariants, toy/known cases, replay, leakage-safe validation, baseline, UQ, sensitivity, and negative results. |
| Under what conditions does the conclusion hold? | Give a conditional decision with risk and alternatives. | Decision dossier, uncertainty, stability region, baselines, and explicit failure conditions. |
| When must the model be redone? | State the support boundary, monitoring triggers, and migration path. | Claim ceiling, extrapolation limits, missing evidence, new-data triggers, and transferable structure. |

These questions define semantic coverage, not compulsory section titles. The
author may choose a conventional venue structure as long as the manifests show
where each obligation is satisfied.

## Architecture

```text
authenticated current S6 snapshot
                |
                v
publication evidence compiler (harness)
read-only sources | claim ceiling | values | reviews | provenance
                |
                v
fresh Codex author (requested gpt-5.6-sol)
abstract.tex | body.tex | caption proposals | evidence anchors
                |
                v
manifest closure (harness)
claims | citations | figures | tables | exact result references
                |
                v
content audit + cold semantic paper reviewer
                |
       suspected scientific inconsistency
                +---------------------> report upstream recovery requirement
                |
                v
content-addressed constrained XeLaTeX build
clean directory | no shell escape | local allowlisted assets | receipt
                |
                v
PDF -> every page rendered to PNG -> every page inspected
                |
                v
cold author-context-isolated layout reviewer
                |
       layout failure
                +---------------------> bounded publication-only revision
                |
                v
content-addressed delivery bundle
PDF | TeX | manifests | logs | reviews | hashes | claim boundary
```

The authoring loop and the scientific graph meet only through a read-only,
authenticated S6 projection. V7.1 reports a scientific inconsistency but does
not revoke an S-stage itself; the existing graph recovery path must perform
that operation. A typography correction stays in the publication layer but
forces re-authoring, content audit, full-page inspection, and a cold
author-context-isolated layout review.

## Responsibility boundary

| Responsibility | Fresh Codex author | Harness code | Independent reviewer |
|---|---:|---:|---:|
| Reader-facing narrative and explanations | Propose | Bind and record | Audit |
| Scientific claims and claim ceiling | Propose wording only | Enforce generic forbidden types and registered bindings | Assess task-specific ceiling from supplied evidence |
| Declared numeric values | Reference typed IDs | Resolve from bound results | Check units, single digits, and meaning |
| Figures and tables | Select admitted IDs and propose captions/layout | Validate, hash-bind, copy/render | Inspect |
| Citations | Select admitted snapshot-bound records | Validate hashes and render | Check relevance/closure |
| LaTeX source | Draft body fragments | Sanitize and render template | Inspect result |
| Compiler and filesystem side effects | None | Execute in constrained build | Observe receipt |
| Scientific gate or external qualification | None | None in publication layer | None |
| Layout acceptance | Revise | Render and record | Decide `APPROVE/REJECT/HUMAN` |

This avoids two symmetrical errors: reducing a frontier model to a rigid
sentence template, and letting persuasive prose become a truth source.

The current schema does not contain a separately typed, task-specific claim
ceiling. Code mechanically enforces generic forbidden claim types and
registered artifact closure, including a conservative forbidden-widening scan
over manuscript and manifest prose; the cold semantic reviewer assesses the
task-specific ceiling from the admitted S0--S6 evidence. Finalization rejects
fixture transports, but these local Codex CLI receipts remain process evidence,
not external or organizational independence.

## Evidence-bound artifacts

Each attempt identity is evidence/request-addressed. Its current authoring
projection may change during the bounded revision loop; role transport history
and content-addressed build directories preserve the receipts for prior work.
The attempt contains:

```text
delivery/paper/v71/attempts/<attempt-id>/
  author_request.json
  evidence_bundle.json
  writer_packet.json
  metadata.json
  source/
    abstract.tex
    body.tex
  manifests/
    claim_ledger.json
    citations.json
    figures.json
    tables.json
  builds/<build-id>/
      main.tex
      main.pdf
      compiler.log
      build_receipt.json
      pages/
        page-001.png
        ...
  reviews/
    content_audit.json
    semantic_review.json
    layout_review.json
  native_roles/
    writer/
    semantic_reviewer/
    layout_reviewer/
  delivery_receipt.json
```

The author writes evidence anchors and typed result references, not final
freehand values. The renderer resolves exact machine-readable values and
frozen snapshot-bound citations. This proves provenance only; metadata and
claim relevance still require the cold semantic review, and no external
citation authority is implied. A cited source snapshot must also be disclosed
in the writer packet so the cold reviewer can compare it. V7.1 requires a
typed JSON citation snapshot and mechanically compares its title, authors,
year, venue, DOI, and URL with the manifest. This binds metadata to the admitted
snapshot, but it does not make the upstream source authoritative. A figure
manifest binds the asset hash,
evidence/claim dependencies, caption, alt text, and, when supplied, a generator
path and hash. A table manifest binds the complete CSV snapshot hash,
evidence/claim dependencies, caption, and row/column bounds. V7.1 does not yet
store a generator command/version or a separate lineage record for every cell.

## Build and layout contract

The provided template uses `ctexart` under XeLaTeX and supports:

- Chinese and Latin text using the required `Noto Serif SC` and
  `Noto Sans SC` font names;
- mathematical notation;
- `graphicx` figures;
- `booktabs` and `tabularx` tables;
- Unicode bookmarks and `hyperref` links;
- restrained headings, captions, margins, and running headers.

The build must:

1. run in a new content-addressed build directory;
2. use `-no-shell-escape`, `-halt-on-error`, and local allowlisted assets;
3. fail if the required fonts are absent and disable MiKTeX automatic package
   installation;
4. reject escaping paths, symlinks, unresolved placeholders, evidence anchors,
   result tokens, unmanifested visuals, and non-snapshot-bound citations;
5. record tool paths, binary hashes, versions, argv hashes, the constrained
   environment hash, compiler logs, and input/output hashes;
6. render every PDF page to a numbered PNG and bind the renderer's resolved
   path, binary hash, version, and argv hash.

The selected tool binaries are recorded after selection, not checked against a
predeclared version allowlist. The current receipt also does not separately
fingerprint installed font files or the TeX package database. Those remain
explicit toolchain prerequisites rather than an attested reproducible-build
image. Pure-read verification does, however, recompute the build ID and argv
hashes, rerun the recorded tool versions, rebuild the TeX twice in a clean
directory, compare the PDF hash, and rerender/compare every page PNG. The
reduced environment, local asset checks, `-no-shell-escape`, and disabled
MiKTeX installer do not establish OS-level network isolation.

Every page must be visually inspected for clipping, overflow, missing glyphs,
broken references, float placement, equation wrapping, table continuation,
figure legibility, bibliography wrapping, whitespace, orphan headings, and page
numbering. The author cannot approve this inspection. A separate layout
reviewer returns a typed `APPROVE`, `REJECT`, or `HUMAN` bound to the final PDF and
page-image hashes through the build hash. The review stores the complete
page-number set and overall findings; it does not require a separate structured
decision for each page.

The end-to-end entry point is:

```powershell
fma-paper run `
  --workspace <task-workspace> `
  --key-file <external-authority-key> `
  --title "<paper title>" `
  --author "<author>" `
  --model gpt-5.6-sol `
  --codex-bin <codex.exe>
```

The receipt records the requested model and always keeps
`served_model_attested=false`; a model name passed to the CLI is not proof of
the served runtime identity.

## Failure and recovery

| Failure | Owner | Recovery |
|---|---|---|
| Unsupported scientific claim or inconsistent result | Publication review, then scientific graph | Stop this publication attempt; use the separate graph recovery path to revoke/rebuild the owning stage when warranted. |
| Missing or unverifiable citation | Upstream source intake | Admit a frozen source snapshot through the source workflow, remove the unsupported claim, or stop. V7.1 does not browse or resolve it. |
| Figure/table does not trace to inputs | Upstream evidence owner | Regenerate and admit a new bound asset/CSV, then create a new publication attempt; never patch pixels or cells manually. |
| XeLaTeX error or unsafe construct | Build harness | Fail closed, repair source/template, and clean-build again. |
| Visual defect | Publication loop | Revise presentation within budget, then rebuild and reinspect every page. |
| Judgment or venue ambiguity | Human boundary | Return `HUMAN`; do not infer acceptance. |

All content, semantic, and layout-driven re-authoring is bounded. Exhausting
the revision budget does not weaken the rubric; it leaves the projection at
`NEEDS_REVISION`. `HUMAN` is reserved for an actual judgment or permission
boundary. Once `delivery_receipt.json` is minted, the attempt is immutable;
repeating the same request verifies and reuses it rather than reopening it.

## Claim boundary

The publication layer can establish:

- traceability from final prose, values, figures, tables, and citations to a
  current S6 evidence snapshot;
- closure of the claims and typed values that the author registered in the
  manifests; exhaustive natural-language claim extraction, units, and
  single-digit literals remain semantic-review responsibilities;
- successful constrained compilation;
- exact artifact/build provenance;
- full-page visual inspection and context-isolated layout acceptance.

It cannot establish:

- that the mathematical model is scientifically correct;
- causal or mechanistic truth;
- external generalization;
- independent scientific qualification;
- venue acceptance or publication;
- authorization for a real-world action.

Unless a separate external authority artifact says otherwise, the publication
bundle fixes:

```text
scientific_qualification_granted = false
real_world_action_authorized = false
```

The decisive design rule is: let the requested frontier model write freely inside
the evidence envelope, and make every consequential paper artifact pass through
code-owned provenance plus cold context-isolated review.
