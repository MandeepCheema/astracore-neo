# Field-Monitoring Database Template

**Document ID:** ASTR-SAFETY-FIELD-MON-V0.1
**Date:** 2026-04-20
**Standard:** ISO 26262-7:2018 §6 (Operation, service, decommissioning) + ISO 26262-8:2018 §6.4.7 (Field-failure feedback)
**Element:** Cross-Party (AstraCore Supplier + Licensee) anomaly intake + tracking
**Status:** v0.1 — first formal release. Closes the placeholder in DIA §9.3 (`docs/safety/dia_template_v0_1.md`). Operationalises SEooC AoU-14 (`docs/safety/seooc_declaration_v0_1.md` §6.4).
**Classification:** Internal (template) — per-engagement instances become Proprietary & Confidential to the named Parties.
**Author:** TBD (Track 2 lead) — currently founder + collaborator
**Reviewer:** TBD (independent reviewer per ISO 26262-2 §7)
**Approver:** TBD (Safety Manager)

---

## 0. Purpose

Per DIA §9, the Licensee reports field anomalies discovered during integration or operation that may be safety-relevant, and both Parties contribute to a shared anomaly database. This template defines:

1. The **database schema** (columns, types, constraints, severity enum)
2. The **intake workflow** (how reports flow from discovery → AstraCore intake)
3. The **triage process** with per-severity SLA
4. **Trend-analysis** queries for safety-case feedback
5. **Retention and access** policies per ISO 26262-8 §10

Per-engagement instances are stored at a Licensee-Supplier shared encrypted location named in the executed DIA §6.2.

### 0.1 Companion documents

- `docs/safety/dia_template_v0_1.md` §9 — contractual anomaly handling
- `docs/safety/seooc_declaration_v0_1.md` §6.4 — AoU-14 (the AoU this template implements)
- `docs/safety/safety_case_v0_1.md` §6 — risk register that field anomalies feed back into

---

## 1. Database schema

### 1.1 Primary record: `anomaly`

| Column | Type | Required? | Notes |
|---|---|:---:|---|
| `anomaly_id` | string (UUID v4) | YES | Generated at intake; immutable |
| `report_date` | ISO 8601 date (UTC) | YES | When Licensee discovered the anomaly |
| `received_date` | ISO 8601 date (UTC) | YES | When Supplier received the report (must be ≤ report_date + 30 days per AoU-14) |
| `licensee_id` | string | YES | The Licensee organisation (anonymised in cross-Party views; full ID in per-Party view) |
| `program_id` | string | YES | Licensee's vehicle program identifier (e.g., "Vehicle Program XYZ rev 2") |
| `fleet_size` | integer | RECOMMENDED | Number of vehicles in scope (denominator for rate calculations) |
| `vehicle_mileage_km` | integer | RECOMMENDED | Cumulative fleet mileage (denominator for rate calculations) |
| `discovery_context` | enum | YES | One of: `integration`, `validation`, `production`, `field_operation`, `recall_followup` |
| `affected_subsystem` | string | YES | RTL module name (`npu_top`, `dms_fusion`, `ecc_secded`, ...) OR SDK component (`astracore.apply`, `tools.npu_ref`, ...) |
| `affected_baseline_sha` | string | YES | Git SHA of the AstraCore baseline at discovery (per DIA §6.3) |
| `symptom` | text | YES | Free-text description of the observed behaviour |
| `repro_steps` | text | RECOMMENDED | Steps to reproduce, if known |
| `workaround_applied` | text | OPTIONAL | If Licensee applied a workaround, what it was |
| `severity` | enum | YES | See §1.2 below |
| `severity_rationale` | text | YES | One-sentence justification for the severity assignment |
| `safety_case_impact` | enum | YES | One of: `none`, `informational`, `requires_re_review`, `requires_safety_advisory` |
| `affected_fmeda_module(s)` | list of strings | OPTIONAL | If the anomaly maps to a FMEDA failure-mode row, name the module(s) |
| `affected_fsr(s)` | list of strings | OPTIONAL | If the anomaly affects an FSR, name them (e.g. `FSR-1.1.3`) |
| `affected_sg(s)` | list of strings | OPTIONAL | Derived from FSRs |
| `triage_owner` | string | YES (set at triage) | Named AstraCore engineer responsible |
| `triage_status` | enum | YES | See §1.4 |
| `triage_initial_response_date` | ISO 8601 date | conditional | Required by triage SLA; SLA depends on severity |
| `root_cause` | text | YES at closure | Filled at root-cause-analysis completion |
| `root_cause_category` | enum | YES at closure | One of: `RTL_defect`, `SDK_defect`, `documentation_defect`, `licensee_misuse`, `out_of_AoU`, `external_factor`, `not_reproducible` |
| `corrective_action` | text | YES at closure | RTL fix / SDK fix / doc update / DIA amendment / etc. |
| `closure_date` | ISO 8601 date | YES at closure | |
| `linked_F4_WP` | string | OPTIONAL | If corrective action becomes an F4 remediation WP, name it |
| `linked_safety_case_risk` | string | OPTIONAL | Links to risk-register row in `safety_case_v0_1.md` §6 (e.g., R8) |

### 1.2 Severity enum (per DIA §9.2)

| Value | Meaning | AstraCore initial response SLA | Root cause + plan SLA |
|---|---|---|---|
| `CRITICAL` | Anomaly that could cause or contribute to a hazard at item level | **1 business day** | 10 business days |
| `WARNING` | Anomaly that degrades a safety mechanism's effectiveness without immediate hazard | 5 business days | 30 business days |
| `INFO` | Anomaly with no direct safety impact (telemetry, performance, doc inaccuracy) | 30 business days | 60 business days |

### 1.3 Discovery-context enum (semantics)

| Value | When applies |
|---|---|
| `integration` | During Licensee's SoC integration, before silicon |
| `validation` | During Licensee's pre-production validation campaigns |
| `production` | During Licensee's production-line testing |
| `field_operation` | After deployment to vehicle / fleet operation |
| `recall_followup` | Discovered as part of investigating a recall or safety advisory |

### 1.4 Triage-status enum

| Value | Meaning |
|---|---|
| `intake` | Received, awaiting initial triage |
| `triaging` | Initial response sent; root-cause investigation under way |
| `awaiting_licensee_info` | Need additional repro / context from Licensee |
| `root_cause_identified` | RCA complete; corrective action being defined |
| `corrective_action_in_progress` | RTL/SDK/doc change being implemented |
| `corrective_action_delivered` | Fix delivered to Licensee on a tagged baseline |
| `closed` | Licensee has acknowledged corrective action; no further work |
| `won't_fix` | Investigation complete; classified as out-of-scope (e.g., AoU violation) |

### 1.5 Safety-case-impact enum

| Value | Meaning | Action required |
|---|---|---|
| `none` | No safety-case rev needed | Log only |
| `informational` | Worth noting in next safety-case rev (background context) | Add to next rev's revision notes |
| `requires_re_review` | Safety case must be re-reviewed before next licensee milestone | Trigger §8 process below |
| `requires_safety_advisory` | Issue safety advisory to all licensees (not just reporter) | Trigger §9 process below |

---

## 2. Intake workflow

### 2.1 Channel

Per DIA §6.2, the channel for safety-relevant anomaly reports is:

- **Email** to AstraCore safety contact named in DIA §1, with `URGENT-SAFETY: <one-line summary>` subject prefix
- **Issue tracker** at the URL named in DIA §6.1 (with restricted access to the named safety contacts)

The Licensee uses a structured intake form (Appendix A) to ensure all required §1.1 columns are populated.

### 2.2 Triggers

The Licensee SHALL report any anomaly meeting any of:

1. **Severity-relevant.** A symptom that could plausibly affect any AoU in SEooC §6 or any FSR in FSC §3
2. **Boundary-signal anomaly.** Any unexpected behaviour observed on a SEooC §2.3 boundary signal (`safe_state_active`, `fault_detected[]`, ECC counters, TMR disagree counter)
3. **Counter-trend anomaly.** Sustained rise in `ecc_corrected_count` rate or `tmr_disagree_count` rate beyond Licensee-defined threshold (typically 2× baseline)
4. **Integration anomaly.** Failure to integrate per Safety Manual §11 verification activities

### 2.3 Acknowledgement

AstraCore acknowledges receipt within **1 business day** for `CRITICAL`, **3 business days** for `WARNING`, **5 business days** for `INFO`. Acknowledgement includes:

- Assigned `anomaly_id`
- Assigned `triage_owner`
- Estimated initial-response date per severity SLA

---

## 3. Triage process

### 3.1 Initial response (per severity SLA in §1.2)

Within the SLA, AstraCore safety contact issues an initial response that includes one of:

| Status | Meaning |
|---|---|
| `acknowledged_investigating` | Severity confirmed; RCA under way |
| `severity_revised` | Severity reclassified after initial review (CRITICAL ↔ WARNING ↔ INFO) — Licensee notified |
| `awaiting_licensee_info` | Need additional repro + context |
| `out_of_scope` | Anomaly is outside SEooC scope (e.g., Licensee SoC bug, item-level design choice) — closure rationale provided |

### 3.2 Root-cause analysis

For `CRITICAL` and `WARNING`, AstraCore performs RCA to one of the §1.1 `root_cause_category` values within the per-severity SLA (§1.2 right column).

If the root cause is `RTL_defect` or `SDK_defect`, a corrective baseline is delivered per DIA §4.3 with the appropriate notification SLA.

### 3.3 Closure

An anomaly closes when:

1. Corrective action delivered to Licensee on a tagged baseline, OR
2. RCA shows the anomaly is `out_of_AoU` (Licensee operating outside the SEooC AoUs), OR
3. RCA shows the anomaly is `not_reproducible` after a reasonable investigation effort, OR
4. Licensee confirms no further action needed

Closure is recorded in the `closure_date` + `corrective_action` columns.

---

## 4. Trend-analysis queries

The shared database supports the following queries (executed quarterly by both Parties' safety managers):

### 4.1 Anomaly rate per affected subsystem (last 90 days)

```sql
SELECT affected_subsystem, COUNT(*) AS n_anomalies, SUM(severity = 'CRITICAL') AS n_critical
FROM anomaly
WHERE received_date >= TODAY - 90
GROUP BY affected_subsystem
ORDER BY n_critical DESC, n_anomalies DESC
```

A subsystem with rising rate may indicate ageing, latent defect, or AoU drift.

### 4.2 Mean-time-to-closure per severity (last 365 days)

```sql
SELECT severity, AVG(closure_date - received_date) AS mttc_days
FROM anomaly
WHERE closure_date IS NOT NULL
  AND received_date >= TODAY - 365
GROUP BY severity
```

If MTTC > SLA, AstraCore safety manager investigates process bottleneck.

### 4.3 Anomalies that triggered safety-case re-review

```sql
SELECT anomaly_id, affected_subsystem, root_cause_category, linked_safety_case_risk
FROM anomaly
WHERE safety_case_impact IN ('requires_re_review', 'requires_safety_advisory')
ORDER BY received_date DESC
```

These feed into Safety Case revision triggers (per `safety_case_v0_1.md` §11.1).

### 4.4 Repeat root-cause categories

```sql
SELECT root_cause_category, COUNT(*) AS n,
       LIST(DISTINCT affected_subsystem) AS subsystems
FROM anomaly
WHERE root_cause IS NOT NULL
  AND received_date >= TODAY - 365
GROUP BY root_cause_category
ORDER BY n DESC
```

A spike in any category (especially `documentation_defect` or `licensee_misuse`) indicates a Safety Manual or AoU clarification opportunity.

### 4.5 AoU-14 compliance check

```sql
SELECT COUNT(*) FILTER (WHERE received_date - report_date > 30) AS late_reports,
       COUNT(*) AS total_reports
FROM anomaly
WHERE received_date >= TODAY - 365
```

`late_reports / total_reports > 5%` triggers DIA §13 amendment to tighten reporting cadence or add Licensee process review.

---

## 5. Retention and access

### 5.1 Retention period

Per ISO 26262-8 §10, all anomaly records retained for **15 years post-product-discontinuation**. Records cannot be deleted; superseded records retain a `superseded_by` link to the new record.

### 5.2 Access roles

| Role | Read | Write | Delete |
|---|:---:|:---:|:---:|
| AstraCore safety contact | All records | All columns | NO (immutable) |
| AstraCore engineer (assigned triage_owner) | All records | Triage + RCA columns of own records | NO |
| AstraCore safety manager (audit) | All records | NO | NO |
| Licensee safety contact | Own records + cross-Licensee aggregate (anonymised) | Initial intake columns of own records | NO |
| Licensee engineer | Own records (read-only) | NO | NO |
| Independent reviewer (per ISO 26262-2 §7) | All records (read-only) | NO | NO |
| TÜV / external assessor | All records (read-only on assessment days) | NO | NO |

### 5.3 Cross-Licensee anonymisation

In any view that aggregates across Licensees, the following columns are anonymised:

- `licensee_id` → hash
- `program_id` → hash
- `fleet_size`, `vehicle_mileage_km` → bucketed (small / medium / large)

Original values remain in per-Licensee views.

### 5.4 Encryption + transport

- At rest: AES-256 encryption on the database storage
- In transit: TLS 1.3 to / from the database
- Backups: encrypted; same retention period as primary

---

## 6. Example records

### 6.1 Example: CRITICAL anomaly

| Column | Value |
|---|---|
| `anomaly_id` | `f47ac10b-58cc-4372-a567-0e02b2c3d479` |
| `report_date` | `2026-09-15` |
| `received_date` | `2026-09-16` |
| `licensee_id` | `L-001` |
| `program_id` | `L-001-vehicle-program-XYZ-rev2` |
| `fleet_size` | 12 (validation fleet) |
| `vehicle_mileage_km` | 24500 |
| `discovery_context` | `validation` |
| `affected_subsystem` | `dms_fusion` |
| `affected_baseline_sha` | `abc123def456` |
| `symptom` | "On 12 of 12 validation vehicles, sustained eyes-closed test (eye_state=CLOSED for 90 frames) does not produce CRITICAL state in `driver_attention_level`. Observed across 47 test runs." |
| `repro_steps` | "Inject eye_state=CLOSED for 90 consecutive gaze_valid pulses; observe driver_attention_level for 100 cycles after injection ends." |
| `severity` | `CRITICAL` |
| `severity_rationale` | "Defeat of FSR-2.2.1; could result in undetected driver drowsiness during deployment." |
| `safety_case_impact` | `requires_re_review` |
| `affected_fmeda_module(s)` | `["dms_fusion"]` |
| `affected_fsr(s)` | `["FSR-2.2.1"]` |
| `affected_sg(s)` | `["SG-2.2"]` |
| `triage_owner` | `engineer-A` |
| `triage_status` | `root_cause_identified` |
| `triage_initial_response_date` | `2026-09-17` (1 business day) |
| `root_cause` | "Off-by-one in cont_closed counter overflow handling at FRAME_64 boundary" |
| `root_cause_category` | `RTL_defect` |
| `corrective_action` | "Patched cont_closed counter overflow logic in `rtl/dms_fusion/dms_fusion.v` line 127. Delivered as baseline `def789abc012`." |
| `closure_date` | `2026-09-25` |
| `linked_F4_WP` | (none — patch is its own WP) |
| `linked_safety_case_risk` | (none — issue closed) |

### 6.2 Example: WARNING anomaly

| Column | Value |
|---|---|
| `anomaly_id` | `c9a8b7d6-...` |
| `report_date` | `2026-10-03` |
| `received_date` | `2026-10-08` |
| `licensee_id` | `L-002` |
| `discovery_context` | `field_operation` |
| `affected_subsystem` | `ecc_secded` |
| `symptom` | "ecc_corrected_count rate has risen 3× over baseline on a subset of fleet vehicles after 2000 hr operation. No double-bit errors; no functional impact." |
| `severity` | `WARNING` |
| `severity_rationale` | "Trend may indicate ageing-correlated SRAM wear; not yet a hazard but warrants investigation before rate exceeds ECC capability." |
| `safety_case_impact` | `informational` |
| `affected_fmeda_module(s)` | `["ecc_secded", "npu_top"]` |
| `triage_owner` | `engineer-B` |
| `triage_status` | `triaging` |
| `triage_initial_response_date` | `2026-10-13` (5 business days) |
| `root_cause` | (TBD — under investigation) |
| `linked_F4_WP` | F4-D-5 (SER analysis re-validation) |

### 6.3 Example: out-of-AoU closure

| Column | Value |
|---|---|
| `anomaly_id` | `b1e2f3a4-...` |
| `report_date` | `2026-11-12` |
| `discovery_context` | `validation` |
| `affected_subsystem` | `dft_isolation_enable` |
| `symptom` | "Unexpected behaviour when `dft_isolation_enable` is asserted high during mission mode. AstraCore appears to enter a degraded-but-functional state." |
| `severity` | `WARNING` (initial), `INFO` (revised) |
| `triage_status` | `won't_fix` |
| `root_cause` | "Licensee asserted `dft_isolation_enable` in mission mode, violating AoU-7 (`docs/safety/seooc_declaration_v0_1.md` §6.2)" |
| `root_cause_category` | `licensee_misuse` |
| `corrective_action` | "Documentation clarification in Safety Manual §11.1 added to make AoU-7 enforcement test mandatory in licensee integration test set. Licensee acknowledged misuse." |
| `closure_date` | `2026-11-20` |

---

## 7. Schema versioning

Schema changes follow DIA §13 amendment process. The current schema is version 1 (`schema_v0_1`). Migrations preserve all historical records.

### 7.1 Backward compatibility

When a new column is added, existing records have `NULL` for that column — they're not retroactively populated. New columns must have explicit default semantics.

When a column's enum is extended, old values remain valid. Removed enum values are migrated to `LEGACY_<value>` to preserve historical traceability.

---

## 8. Safety-case re-review process (triggered by `safety_case_impact = requires_re_review`)

Within 30 business days of the anomaly's closure:

1. AstraCore safety manager reviews the affected SG / FSR / TSR / mechanism in the safety case
2. If material change to any of: aggregate FMEDA, mechanism DC, AoU set, ITs that reference the affected module → trigger a new revision of the affected work product
3. Update `safety_case_v0_1.md` §6 risk register with a new row pointing to the anomaly
4. Notify all current Licensees per DIA §4.3 if the change is `requires_safety_advisory`

---

## 9. Safety-advisory process (triggered by `safety_case_impact = requires_safety_advisory`)

Within 5 business days of the anomaly's closure:

1. AstraCore safety manager drafts a safety advisory bulletin
2. Advisory issued to **all current Licensees** (not just the reporting Licensee), even if they have not encountered the anomaly
3. Bulletin includes: affected baseline SHAs, symptom, recommended Licensee action, corrected-baseline SHA (if available)
4. Acknowledgement required from each Licensee within 30 business days
5. Logged as a `safety_advisory_id` in the database, cross-referenced from the anomaly record

---

## 10. Open items for v0.2

1. **Database technology choice** — current template is schema-only; v0.2 selects a backing store (proposed: encrypted SQLite for single-Party + PostgreSQL for cross-Party shared)
2. **Web-form intake** — current process is email + issue tracker; v0.2 adds a web form that pre-validates the §1.1 required columns
3. **Automated trend alerts** — §4 queries today are manual quarterly; v0.2 adds automated alerts (e.g., "any subsystem with >3 CRITICAL in 30 days → page safety manager")
4. **Anonymisation function** — §5.3 specifies the policy; v0.2 specifies the hash function + key management
5. **Named Track 2 lead, independent reviewer, Safety Manager / Approver** — currently TBD per §0
6. **Linkage to F4-IT-* test plan** — when an anomaly's RCA identifies a missing IT, automatically open the corresponding IT WP

---

## 11. Document control

| Field | Value |
|---|---|
| Document ID | ASTR-SAFETY-FIELD-MON-V0.1 |
| Revision | 0.1 |
| Revision date | 2026-04-20 |
| Author | TBD (Track 2 lead) |
| Reviewer | TBD |
| Approver | TBD |
| Distribution | Internal (template) + per-engagement DIA Parties |
| Retention | 15 years post-product-discontinuation per ISO 26262-8 §10 |
| Supersedes | None — this is v0.1; closes DIA §9.3 placeholder |

### 11.1 Revision triggers

This template is re-issued on any of:

1. New required field added to §1.1 schema
2. New severity level added to §1.2 (today: 3 levels) or new SLA negotiated
3. New trend-analysis query added to §4 that becomes a standard quarterly review
4. New retention requirement (e.g., from updated ISO 26262 edition or regional regulation)
5. New access role added (e.g., regulator with read-only access)
6. Confirmation review feedback that changes any clause

---

## Appendix A — Licensee intake form template

```
ANOMALY REPORT — AstraCore Neo IP

Required (do not submit without these):
  Date discovered:           [YYYY-MM-DD]
  Licensee organisation:     [name]
  Vehicle program:           [identifier]
  Discovery context:         [integration / validation / production / field_operation / recall_followup]
  Affected subsystem:        [npu_top / dms_fusion / ecc_secded / safe_state_controller / plausibility_checker / tmr_voter / astracore.apply / ...]
  Affected baseline SHA:     [git SHA]
  Symptom:                   [free-text — what was observed?]
  Severity (your assessment): [CRITICAL / WARNING / INFO]
  Severity rationale:        [one sentence]

Recommended (improves triage speed):
  Fleet size in scope:       [number]
  Cumulative fleet mileage:  [km]
  Repro steps:               [free-text]
  Workaround applied:        [free-text]
  Mapping to FSR / FMEDA:    [if known]

Submission:
  Email:    [astracore safety contact per DIA §1] with subject prefix [URGENT-SAFETY: <one-line summary>]
  Tracker:  [URL per DIA §6.1]
```
