import csv
import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Tuple


FINDINGS_SPECIAL = "[Findings]"
CONCLUSION_SPECIAL = "[Conclusion]"
TEXT_COLUMN = "text"
CSV_COLUMNS = [
    "patient_id",
    "accession_number",
    "study_date",
    "source_row",
    "findings",
    "conclusion",
    TEXT_COLUMN,
]
PAIR_PATTERN = re.compile(r"\[Findings\](.*?)\[Conclusion\](.*)")


@dataclass(frozen=True)
class ReportRecord:
    patient_id: str
    accession_number: str
    study_date: str
    source_row: str
    findings: str
    conclusion: str
    text: str


def _clean_value(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return " ".join(text.split())


def _first_present(row: Mapping, names: Iterable[str]) -> str:
    for name in names:
        if name in row:
            value = _clean_value(row[name])
            if value:
                return value
    return ""


def format_report_text(findings: str, conclusion: str) -> str:
    return f"{FINDINGS_SPECIAL} {findings} {CONCLUSION_SPECIAL} {conclusion}"


def build_report_record(row: Mapping, source_row: object = "") -> ReportRecord:
    findings = _clean_value(row.get("Findings", row.get("findings", "")))
    conclusion = _clean_value(row.get("Conclusion", row.get("conclusion", "")))
    return ReportRecord(
        patient_id=_first_present(row, ("PatientID", "patient_id", "patientId")),
        accession_number=_first_present(
            row, ("accession_number", "AccessionNumber", "accessionNumber")
        ),
        study_date=_first_present(row, ("StudyDate", "study_date", "studyDate")),
        source_row=_clean_value(source_row),
        findings=findings,
        conclusion=conclusion,
        text=format_report_text(findings, conclusion),
    )


def write_report_split_csv(path: str, rows) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for source_row, row in rows.iterrows():
            record = build_report_record(row, source_row=source_row)
            writer.writerow(record.__dict__)


def _looks_like_report_csv(first_line: str) -> bool:
    if not first_line:
        return False
    columns = {name.strip().lower() for name in next(csv.reader([first_line]))}
    return TEXT_COLUMN in columns or {"findings", "conclusion"}.issubset(columns)


def _parse_text_pair(text: str) -> Tuple[str, str]:
    match = PAIR_PATTERN.search(text.strip())
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def load_report_records(path: str) -> List[ReportRecord]:
    records: List[ReportRecord] = []
    with open(path, "r", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        if _looks_like_report_csv(first_line):
            reader = csv.DictReader(f)
            for row in reader:
                record = build_report_record(row, source_row=row.get("source_row", ""))
                if not (record.findings and record.conclusion):
                    findings, conclusion = _parse_text_pair(record.text or row.get(TEXT_COLUMN, ""))
                    if findings and conclusion:
                        record = ReportRecord(
                            patient_id=record.patient_id,
                            accession_number=record.accession_number,
                            study_date=record.study_date,
                            source_row=record.source_row,
                            findings=findings,
                            conclusion=conclusion,
                            text=format_report_text(findings, conclusion),
                        )
                if record.findings and record.conclusion:
                    records.append(record)
            return records

        for line_number, line in enumerate(f):
            findings, conclusion = _parse_text_pair(line)
            if findings and conclusion:
                records.append(
                    ReportRecord(
                        patient_id="",
                        accession_number="",
                        study_date="",
                        source_row=str(line_number),
                        findings=findings,
                        conclusion=conclusion,
                        text=format_report_text(findings, conclusion),
                    )
                )
    return records


def load_findings_conclusion_pairs(path: str) -> List[Tuple[str, str]]:
    return [(record.findings, record.conclusion) for record in load_report_records(path)]
