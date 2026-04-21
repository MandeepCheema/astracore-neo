# Development Interface Agreement (DIA) — Template

**Document ID:** ASTR-SAFETY-DIA-TEMPLATE-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-8:2018 §5 (Interfaces within distributed developments) + ISO 26262-10:2018 §6 (Distributed development with multiple suppliers)
**Status:** v0.1 — TEMPLATE for licensee-specific instantiation. Per-engagement DIAs derive from this template by filling `[PLACEHOLDER]` fields.
**Classification:** Internal until executed; on signature becomes Proprietary & Confidential to the named parties.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer:** TBD (independent reviewer per ISO 26262-2 §7) + AstraCore legal counsel
**Approver:** TBD (founder + Safety Manager)

---

## 0. How to use this template

This is a **template**. Per executed DIA, replace every `[PLACEHOLDER]` field with the licensee-specific value, fill the §3 lifecycle table per the licensee's program plan, and route through legal counsel before signature. Do not modify the safety-relevant clauses (§4, §5, §7, §9) without re-running the risk assessment in §12.

Each clause that **MUST** appear in every executed DIA is marked **[REQUIRED]**. Clauses marked **[OPTIONAL]** may be removed if not relevant to the engagement.

### 0.1 Companion documents

- `docs/safety/seooc_declaration_v0_1.md` §6.4 AoU-13 — the AoU this DIA satisfies
- `docs/safety/safety_manual_v0_5.md` §10 — denormalized AoU index this DIA enforces
- `docs/safety/iso26262_gap_analysis_v0_1.md` Part 8 §5 row — gap this template closes
- `docs/safety/findings_remediation_plan_v0_1.md` — F4 work packages that may impact licensee timeline
- `docs/spec_sheet_rev_1_4.md` — IP datasheet defining the licensed scope

---

## 1. Parties

**[REQUIRED]**

This Development Interface Agreement ("DIA") is entered into as of `[EXECUTION DATE]` between:

**SUPPLIER:** Astracore AI Technologies Private Limited
- Registered office: `[ASTRACORE REGISTERED ADDRESS]`
- Safety contact: `[ASTRACORE SAFETY MANAGER]` ( `[email]` )
- Legal contact: `[ASTRACORE LEGAL COUNSEL]` ( `[email]` )

**LICENSEE:** `[LICENSEE LEGAL NAME]`
- Registered office: `[LICENSEE ADDRESS]`
- Safety contact: `[LICENSEE FUNCTIONAL SAFETY MANAGER]` ( `[email]` )
- Legal contact: `[LICENSEE LEGAL COUNSEL]` ( `[email]` )
- Item identification: `[LICENSEE ITEM NAME — e.g., "Acme Vehicle Program XYZ ECU revision 2"]`

Hereafter referred to collectively as "the Parties".

---

## 2. Purpose and scope

**[REQUIRED]**

### 2.1 Purpose

This DIA defines the safety lifecycle activities, work products, and interfaces between the Supplier (AstraCore) and the Licensee for the integration of the AstraCore Neo NPU + Sensor-Fusion IP block ("the IP") into the Licensee's item ("the Item"). It satisfies ISO 26262-8 §5 and ISO 26262-10 §6 requirements for distributed development.

### 2.2 In scope

- Delivery of the IP and associated safety-case documentation per `docs/safety/seooc_declaration_v0_1.md` §2.1
- Joint reviews of safety-relevant analyses per §7 below
- Change management on the IP per §8 below
- Field anomaly handling per §9 below

### 2.3 Out of scope (explicit)

- Item-level HARA on the Licensee's vehicle program (Licensee responsibility per AoU-15)
- Licensee SoC integration including memory PHY, package, supervisor MCU, clock distribution (Licensee responsibility)
- Vehicle-level safety case (vehicle OEM responsibility, downstream of Licensee)
- The full list of Licensee-supplied items per `docs/spec_sheet_rev_1_4.md` §12

---

## 3. Lifecycle phase allocation

**[REQUIRED]**

Per ISO 26262-8 §5.4.4, the following table allocates each ISO 26262 lifecycle phase activity to one of the Parties (S = Supplier, L = Licensee, J = Joint).

| ISO 26262 phase | Activity | Allocation |
|---|---|:---:|
| Part 2 §5 (Safety culture) | Maintain safety culture per signed-off safety policy | S + L (each at their org) |
| Part 2 §6 (FSM during development) | Functional Safety Management for the IP block | S |
| Part 2 §6 (FSM during development) | Functional Safety Management for the Item | L |
| Part 2 §7 (Confirmation measures) | Independent review / audit of IP block work products | S (with optional Licensee observer) |
| Part 2 §7 (Confirmation measures) | Independent review / audit of Item work products | L |
| Part 3 §5 (Item definition) | Item definition for the assumed item context | S (`hara_v0_1.md` §3) |
| Part 3 §5 (Item definition) | Item definition for the actual Licensee item | L |
| Part 3 §6 (HARA) | Assumed HARA on three reference use cases | S (`hara_v0_1.md`) |
| Part 3 §6 (HARA) | Item-level HARA on Licensee's vehicle program | L |
| Part 3 §7 (Functional Safety Concept) | Assumed FSC + 25 FSRs | S (`functional_safety_concept_v0_1.md`) |
| Part 3 §7 (Functional Safety Concept) | Item-level FSC | L (may revise FSRs from S) |
| Part 4 §6 (Technical Safety Concept) | TSC for IP block | S (planned W12-W14) |
| Part 4 §6 (Technical Safety Concept) | TSC for Item-level integration | L |
| Part 4 §7 (System architectural design) | IP block architecture | S |
| Part 4 §7 (System architectural design) | Item architecture (incl. IP block placement) | L |
| Part 5 §7 (HW safety requirements) | IP block HW safety requirements | S |
| Part 5 §7.4.5 (FMEDA) | FMEDA on IP block modules | S (`docs/safety/fmeda/`) |
| Part 5 §7.4.5 (FMEDA) | Item-level FMEDA roll-up | L |
| Part 5 §11 (HW integration & testing) | Fault-injection campaigns on IP RTL (in WSL) | S (`docs/safety/fault_injection/`) |
| Part 5 §11 (HW integration & testing) | Fault-injection on integrated SoC | L |
| Part 5 §11 (HW integration & testing) | Post-silicon fault injection at Licensee tape-out | L |
| Part 6 §7 (SW safety requirements) | SW requirements for AstraCore SDK | S |
| Part 6 §10 (SW unit verification) | SDK unit + integration tests (1352 today) | S |
| Part 7 (Production + operation) | Production planning | L |
| Part 7 (Production + operation) | Field monitoring + anomaly intake | J (per §9 below) |
| Part 8 §5 (DIA) | This document; maintenance | J |
| Part 8 §6 (Safety case) | IP block safety case | S |
| Part 8 §6 (Safety case) | Item-level safety case | L |
| Part 8 §7 (Configuration management) | IP block CM (git, baselines) | S |
| Part 8 §7 (Configuration management) | Item-level CM (incl. delivered IP baselines) | L |
| Part 8 §8 (Change management) | Change-impact analysis on IP changes | S (with notification per §8) |
| Part 8 §11 (Tool qualification) | TCL evaluation of S's tools | S (`tcl_evaluations_v0_1.md`) |
| Part 8 §11 (Tool qualification) | TCL evaluation of L's tools (incl. licensee synthesis flow) | L |
| Part 8 §13 (Hardware component qualification) | OpenTitan + memory PHY + foundry SRAM compiler | L (with S advisory) |
| Part 9 §5 (ASIL decomposition) | IP block decomposition options (lockstep, dual-rail) | S advisory |
| Part 9 §5 (ASIL decomposition) | Item-level ASIL decomposition decision | L |
| Part 9 §7 (CCF analysis) | IP block CCF analysis (W14-W15 planned) | S |
| Part 9 §7 (CCF analysis) | Item-level CCF (incl. multiple instances) | L |
| Part 11 §4.6 (SEooC declaration) | SEooC for IP block | S (`seooc_declaration_v0_1.md`) |
| Part 11 §4.7 (Assumptions of use) | AoUs declared in SEooC §6 | S (Licensee accepts via signature) |
| Part 11 §4.7 (Assumptions of use) | AoU verification | L |
| Part 11 §7 (Soft errors) | Soft Error Rate analysis on IP block (planned W10) | S |
| Part 11 §7 (Soft errors) | SER projection on Licensee silicon | L (with S input) |

`[OPTIONAL]` Licensee may add per-program activities by appending rows; new rows shall not contradict any in this template without DIA amendment per §13.

---

## 4. Work products delivered by Supplier to Licensee

**[REQUIRED]**

### 4.1 Initial delivery (at DIA signature)

| Deliverable | Format | Reference |
|---|---|---|
| Spec sheet rev `[VERSION AT SIGNATURE]` | PDF + markdown | `docs/spec_sheet_rev_1_4.md` |
| SEooC declaration | Markdown + signed cover sheet | `docs/safety/seooc_declaration_v0_1.md` |
| HARA | Markdown | `docs/safety/hara_v0_1.md` |
| Functional Safety Concept + FSRs | Markdown | `docs/safety/functional_safety_concept_v0_1.md` |
| Safety Manual | Markdown | `docs/safety/safety_manual_v0_5.md` (or current revision) |
| TCL evaluations | Markdown | `docs/safety/tcl/tcl_evaluations_v0_1.md` |
| Findings remediation plan | Markdown | `docs/safety/findings_remediation_plan_v0_1.md` |
| FMEDA reports for catalogued modules | Markdown + JSON baseline | `docs/safety/fmeda/` + `baseline.json` |
| Fault-injection campaign manifests | YAML + report templates | `sim/fault_injection/campaigns/` + `docs/safety/fault_injection/` |
| RTL source bundle | Tagged git baseline | `rtl/` at the SHA recorded in §6.3 |
| SDK source bundle | Tagged git baseline | `astracore/`, `tools/`, `tests/` at the same SHA |
| Build + test instructions | Markdown | `README.md` + `docs/customer_integration_guide.md` |

`[REQUIRED]` Each deliverable is accompanied by SHA-256 manifest per Safety Manual §2.3. Licensee verifies on receipt.

### 4.2 Recurring deliveries (during integration phase)

| Deliverable | Cadence | Format |
|---|---|---|
| Updated FMEDA / fault-injection reports as F4 phases land | Monthly OR on milestone close | Markdown + baseline JSON delta |
| Updated RTL baselines | Quarterly OR on F4 phase close | Tagged git release |
| Updated spec sheet rev | On any change to a §13.1 revision-trigger condition | PDF + markdown diff |
| Updated TCL evaluations | On tool addition / version bump (per `tcl_evaluations_v0_1.md` §7) | Markdown |
| Updated Safety Manual | On §13.2 revision-trigger | Markdown |
| Field anomaly database snapshot | Quarterly | CSV / JSON |

### 4.3 Notification deliveries (event-driven)

| Trigger | Notification SLA |
|---|---|
| Discovered safety-relevant defect in delivered RTL or SDK | Within 5 business days of confirmation |
| FMEDA regression > 1 pp SPFM/LFM drop or > 0.001 FIT PMHF rise on any module | Within 10 business days |
| New ASR added to SEooC §4.1 | Within 10 business days |
| New AoU added to SEooC §6 | **Within 1 business day** (AoUs change Licensee responsibilities) |
| Tool qualification status change (TCL2/TCL3 closure or new TCL2/TCL3 finding) | Within 10 business days |
| Confirmation review independent reviewer change | Within 30 business days |

---

## 5. Work products delivered by Licensee to Supplier

**[REQUIRED]**

### 5.1 At DIA signature

| Deliverable | Reference |
|---|---|
| Item identification per §1 | DIA cover page |
| Licensee functional-safety manager + escalation contacts | DIA §1 |
| Licensee program timeline + integration milestones | `[ATTACHMENT A]` |
| Licensee assumed-ASIL claim per HARA on the actual vehicle program (if available; preliminary acceptable) | `[ATTACHMENT B]` |
| Licensee tool list (synthesis flow, formal tools, target node PDK) | `[ATTACHMENT C]` |

### 5.2 During integration

| Deliverable | Cadence |
|---|---|
| Item-level HARA per AoU-15 | One-time, before integration test sign-off |
| Item-level FSC + TSC | One-time, before integration test sign-off |
| Integration test results per Safety Manual §11 | At each milestone gate |
| AoU verification reports (per §10 of Safety Manual; one row per AoU-1..17) | One-time before production |
| Field anomaly reports per AoU-14 | Within 30 days of discovery |
| Confirmation review records (independent reviewer named per §7 below) | At each milestone gate |

---

## 6. Interface for exchange

**[REQUIRED]**

### 6.1 Technical channels

| Purpose | Channel | Authority |
|---|---|---|
| Source code + safety docs delivery | Private git repository on `[GIT HOST — e.g., GitHub Enterprise instance]` with Licensee-specific branch | AstraCore engineering |
| Updates / patches | Same repository; tagged releases + signed commits | AstraCore engineering |
| Bug reports / defect tracking | `[ISSUE TRACKER URL]` with restricted access | Both Parties |
| Confidential communications | Encrypted email (PGP keys exchanged at signature) OR shared Signal group | Safety contacts named in §1 |
| Audio / video meetings | Encrypted conference platform `[ZOOM / TEAMS / MEET]` with end-to-end encryption | Both Parties |

### 6.2 Administrative channels

| Purpose | Channel |
|---|---|
| DIA amendments | Written, signed by both Parties' authorised representatives |
| Notification of safety-relevant defects | Email to safety contacts in §1 with `URGENT-SAFETY` subject prefix |
| Quarterly review minutes | Shared document in `[DOC PLATFORM]` accessible to both Parties' safety teams |
| Field anomaly database | Shared encrypted spreadsheet on `[DOC PLATFORM]` |

### 6.3 Configuration baseline reference

The IP baseline as of DIA execution is identified by:

- AstraCore git SHA: `[BASELINE SHA — fill at execution]`
- Spec sheet revision: `[REV NUMBER]`
- Safety case revision: `[CONSOLIDATED REV — e.g., "Safety Case v0.5 bundle 2026-04-20"]`
- SHA-256 manifest: `[ATTACHMENT D — manifest file]`

`[REQUIRED]` Licensee verifies the SHA-256 manifest within 10 business days of receipt.

---

## 7. Joint reviews

**[REQUIRED]**

### 7.1 Cadence

| Review type | Frequency | Participants | Output |
|---|---|---|---|
| Operational status sync | Monthly | S + L technical leads | Action log |
| Safety review | Quarterly | S + L safety managers + independent reviewer | Signed minutes + action log |
| Milestone gate review | Per Licensee program milestone | S + L safety managers + reviewer + Licensee program manager | Signed gate-pass / gate-fail decision |
| Confirmation review (per ISO 26262-2 §7) | At each safety case revision | Independent reviewer named per §7.2 | Signed confirmation review report |

### 7.2 Independent reviewer

`[REQUIRED]` ISO 26262-2 §7 confirmation measures require an independent reviewer for any work product associated with ASIL ≥ B.

For this DIA, the independent reviewer is:

- **Name / organisation:** `[REVIEWER NAME / FIRM]`
- **Independence basis:** `[ORGANISATIONAL / FINANCIAL]` independent from both S and L
- **Engagement model:** `[NAMED CONSULTANT / TÜV-CERTIFIED FIRM / etc.]`

`[OPTIONAL]` AstraCore proposes TÜV SÜD India as the default independent reviewer; pre-engagement workshop scheduled W6 per `docs/best_in_class_design.md` §7.2.

### 7.3 Escalation

If a review identifies a safety-relevant disagreement between the Parties:

1. Try to resolve at the safety-manager level within 5 business days
2. If unresolved, escalate to the founder (S) + program executive (L) within 10 business days
3. If still unresolved, the independent reviewer's written opinion is binding for safety-case purposes (commercial implications handled separately per the master license agreement)

---

## 8. Change management

**[REQUIRED]**

### 8.1 Change classification

| Class | Definition | Notification SLA | Approval |
|---|---|---|---|
| **Class A — Safety-relevant** | Changes that may alter the FMEDA / fault-injection / SEooC AoU / HARA / FSC | 5 business days | Joint sign-off (S + L safety managers) |
| **Class B — IP-functional** | Changes to RTL or SDK that don't change safety mechanisms or boundary signals | Quarterly delivery | S sign-off; L review on next delivery |
| **Class C — Documentation** | Doc rev bumps, typo fixes, clarifications | Quarterly delivery | S sign-off |
| **Class D — Tooling** | Tool version bumps within a major release | Annual delivery | S sign-off |

### 8.2 Change-impact analysis

`[REQUIRED]` All Class A changes by AstraCore are accompanied by:

1. Change description
2. Affected RTL files + SDK files + safety docs
3. FMEDA delta (re-run on affected modules; baseline comparison)
4. Fault-injection re-run requirement (if any campaigns affected)
5. SEooC AoU impact (if any)
6. HARA impact (if any)
7. Recommended Licensee re-verification scope

Licensee acknowledges receipt + plans re-verification within the agreed timeline.

### 8.3 Licensee changes

`[REQUIRED]` Per AoU-12, Licensee shall not modify AstraCore RTL or Python toolchain without:

1. Re-running the safety verification suite (`pytest` + WSL cocotb gate)
2. Re-running affected FMEDAs (`python -m tools.safety.regress_check`)
3. Updating Licensee item-level safety case
4. Notifying Supplier of the modification scope (informational; does not require S approval)

Modifications that re-distribute the IP outside the Licensee's organisation require DIA amendment per §13.

---

## 9. Field anomaly handling

**[REQUIRED]**

### 9.1 Reporting (per AoU-14)

`[REQUIRED]` Licensee reports any anomaly discovered during integration or operation that may be safety-relevant within **30 days** of discovery.

Report format:

| Field | Description |
|---|---|
| Date of discovery | YYYY-MM-DD |
| Vehicle / fleet identification | `[Licensee fleet ID]` |
| Operating context | Mileage, environment, mission profile |
| Symptom | Free-text description |
| Affected AstraCore subsystem | Module name from `rtl/` or `astracore/` |
| Severity (per Licensee classification) | INFO / WARNING / CRITICAL |
| Repro instructions | If known |
| Workaround applied | If any |

Reports go to the AstraCore safety contact named in §1 with `URGENT-SAFETY` subject prefix.

### 9.2 Triage SLA

| Severity | AstraCore initial response | Root cause + plan |
|---|---|---|
| CRITICAL | 1 business day | 10 business days |
| WARNING | 5 business days | 30 business days |
| INFO | 30 business days | 60 business days |

### 9.3 Field anomaly database

Both Parties contribute to a shared anomaly database (per §6.2). Anonymised aggregate trends inform future safety case revisions.

**Schema + workflow specified in `docs/safety/field_monitoring_template_v0_1.md`** — defines the 23-column anomaly record (§1.1), severity enum + SLA (§1.2), intake workflow (§2), triage process (§3), 5 standard trend-analysis queries (§4), retention + access roles (§5), example records (§6), schema versioning (§7), safety-case re-review process (§8), safety-advisory process (§9), and Licensee intake form template (Appendix A).

### 9.4 Recall / safety advisory

If a field anomaly triggers a recall or safety advisory:

1. Joint root-cause analysis within 30 business days of advisory issuance
2. AstraCore provides a corrective baseline (RTL / SDK / safety doc updates) within the joint plan
3. Licensee handles vehicle-level recall logistics

---

## 10. Confidentiality and IP protection

**[REQUIRED]**

### 10.1 Confidentiality

Both Parties agree to keep the contents of this DIA, the IP source, and all safety documentation confidential per the master license agreement. Material marked **Proprietary & Confidential** in any companion document remains confidential after DIA termination.

### 10.2 IP ownership

- AstraCore retains ownership of the IP and all derivative works
- Licensee retains ownership of their integrated SoC + Item-level safety case

### 10.3 Right to inspect

Each Party has the right to inspect the other's safety case work products on reasonable notice (≥ 10 business days), limited to the scope of activities allocated in §3.

---

## 11. Term and termination

**[REQUIRED]**

### 11.1 Term

This DIA is effective from `[EXECUTION DATE]` and continues until:

- The licensed product reaches end-of-life (EOL) per the master license agreement, OR
- Termination per §11.2 below

### 11.2 Termination

Either Party may terminate this DIA on 90 days' written notice, subject to the long-tail safety obligations in §11.3.

### 11.3 Long-tail obligations

Per ISO 26262-7 §6, field-monitoring obligations continue for **15 years post-product-discontinuation**. Both Parties commit to:

- Maintain field anomaly intake channels
- Preserve safety case documents per the retention period
- Respond to safety-relevant inquiries within the SLA in §9 even post-termination

---

## 12. Risk acknowledgement

**[REQUIRED]**

### 12.1 Known limitations of the IP at DIA execution

The Parties acknowledge that, at the date of DIA execution, the IP has the following safety-relevant gaps documented in `docs/safety/`:

- **F4 remediation plan** open (`docs/safety/findings_remediation_plan_v0_1.md`); Phase A / B / C / D in flight
- **npu_top SPFM 2.08 %** today vs ASIL-B 90 % target — F4-A Phase A planned to lift to ~75 %
- **safe_state_controller F4-A-7 (TMR on 2-bit FSM)** identified as MUST FIX before ASIL-B safety case v1.0
- **ECC SECDED layout limitation** (parity-bit aliasing on data positions {0,1,3,7,15,31,63}) — F4-D-6 closes
- **Crypto RTL pending** Track 3 OpenTitan integration (W4-W7)
- **Clock monitor RTL pending** (gap noted in SEooC §5)
- **C++ runtime pending** (F1-B3)
- **3 tools require formal qualification** (Yosys F4-B-7, OpenROAD F4-D-7, ASAP7 F4-D-8)

### 12.2 Licensee acceptance

By signing, Licensee acknowledges:

1. The IP is licensed **as-is at the baseline SHA** named in §6.3
2. The remediation plan timeline is a roadmap; Supplier commits to good-faith execution but does not warrant delivery dates
3. The Licensee's item-level safety case must accommodate the limitations in §12.1 OR await the corresponding F4 phase closure
4. The assumed HARA in `docs/safety/hara_v0_1.md` is a starting point; Licensee performs item-level HARA per AoU-15 and may revise

---

## 13. DIA amendment + revision

**[REQUIRED]**

### 13.1 Amendment process

DIA amendments require:

1. Written proposal by either Party
2. Review by both safety managers within 30 business days
3. Independent reviewer concurrence (per §7.2) for any safety-relevant change
4. Authorised-representative signatures from both Parties
5. Distribution per §6.2

### 13.2 Mandatory re-amendment triggers

This DIA must be amended on any of:

1. Change of independent reviewer (per §7.2)
2. Change of either safety manager (per §1)
3. SEooC §6.5 AoU set materially changes (denormalised in Safety Manual §10)
4. Spec-sheet rev that changes the IP scope of license
5. Tool qualification activity completes that resolves a §12.1 risk
6. Field anomaly that materially changes the §9 triage SLA expectations
7. Either Party's organisational restructuring that affects §3 lifecycle allocation
8. AstraCore revision of this template (i.e., a future ASTR-SAFETY-DIA-TEMPLATE-V0.2+) — existing executed DIAs remain on the executed revision unless re-signed

### 13.3 Template revision history

| Template rev | Date | Changes |
|---|---|---|
| v0.1 | 2026-04-20 | Initial release; based on ISO 26262-8 §5 + ISO 26262-10 §6 + AoU-13 |

---

## 14. Signatures

**[REQUIRED at execution]**

By signature below, the named representatives acknowledge they have authority to bind their respective organisations to this DIA.

**For AstraCore:**

Name: `[ASTRACORE AUTHORISED REPRESENTATIVE]`
Title: `[TITLE]`
Signature: ____________________
Date: `[YYYY-MM-DD]`

**For Licensee:**

Name: `[LICENSEE AUTHORISED REPRESENTATIVE]`
Title: `[TITLE]`
Signature: ____________________
Date: `[YYYY-MM-DD]`

**Witness (Independent reviewer per §7.2):**

Name: `[REVIEWER NAME]`
Organisation: `[FIRM]`
Signature: ____________________
Date: `[YYYY-MM-DD]`

---

## Appendix A — Licensee program timeline + integration milestones

`[ATTACHMENT — to be provided by Licensee at DIA execution]`

## Appendix B — Licensee preliminary item-level HARA

`[ATTACHMENT — Licensee's HARA on the actual vehicle program; preliminary acceptable]`

## Appendix C — Licensee tool list

`[ATTACHMENT — Licensee's synthesis flow + formal tools + target node PDK]`

## Appendix D — IP baseline SHA-256 manifest

`[ATTACHMENT — manifest file generated at delivery]`

---

## Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-DIA-TEMPLATE-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) — currently founder + collaborator + AstraCore legal counsel |
| Reviewer | TBD |
| Approver | TBD (founder + Safety Manager) |
| Distribution | Internal only at template stage; per-engagement DIAs distributed to executed Parties |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1 of the template |
