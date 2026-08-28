import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExternalDatasetError(ValueError):
    """Raised when an upstream benchmark artifact has an unsupported shape."""


class ExternalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QasperReference(ExternalModel):
    answer: str
    answer_type: Literal["extractive", "abstractive", "boolean", "none"]
    evidence: list[str]


class QasperCase(ExternalModel):
    paper_id: str
    title: str
    question_id: str
    question: str
    source_split: str
    references: list[QasperReference] = Field(min_length=1)


class SciFactRationale(ExternalModel):
    doc_id: int
    label: Literal["SUPPORT", "CONTRADICT"]
    sentence_indices: list[int] = Field(min_length=1)
    sentences: list[str] = Field(min_length=1)


class SciFactDocument(ExternalModel):
    doc_id: int
    title: str
    abstract: list[str] = Field(min_length=1)


class SciFactCase(ExternalModel):
    claim_id: int
    claim: str
    source_split: str
    label: Literal["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
    cited_doc_ids: list[int]
    documents: list[SciFactDocument]
    rationales: list[SciFactRationale]


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalDatasetError(f"Could not load {path}: {exc}") from exc


def _qasper_reference(answer: dict, *, text_evidence_only: bool) -> QasperReference:
    evidence = answer.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ExternalDatasetError("QASPER answer evidence must be a list of strings.")
    if text_evidence_only:
        evidence = [item for item in evidence if "FLOAT SELECTED" not in item]
    if answer.get("unanswerable"):
        return QasperReference(answer="Unanswerable", answer_type="none", evidence=[])
    spans = answer.get("extractive_spans") or []
    if spans:
        return QasperReference(
            answer=", ".join(spans), answer_type="extractive", evidence=evidence
        )
    if answer.get("free_form_answer"):
        return QasperReference(
            answer=answer["free_form_answer"], answer_type="abstractive", evidence=evidence
        )
    if answer.get("yes_no") is True:
        return QasperReference(answer="Yes", answer_type="boolean", evidence=evidence)
    if answer.get("yes_no") is False:
        return QasperReference(answer="No", answer_type="boolean", evidence=evidence)
    raise ExternalDatasetError("QASPER annotation does not contain a supported answer type.")


def load_qasper(
    path: str | Path,
    *,
    source_split: str,
    text_evidence_only: bool = True,
    limit: int | None = None,
) -> list[QasperCase]:
    """Read native QASPER JSON without changing its official answer/evidence semantics."""
    payload = _read_json(Path(path))
    if not isinstance(payload, dict):
        raise ExternalDatasetError("QASPER root must be an object keyed by paper ID.")
    cases = []
    for paper_id, paper in payload.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("qas"), list):
            raise ExternalDatasetError(f"Invalid QASPER paper entry: {paper_id}")
        for qa in paper["qas"]:
            answers = qa.get("answers")
            if not isinstance(answers, list) or not answers:
                raise ExternalDatasetError(f"QASPER question {qa.get('question_id')} has no answers.")
            references = []
            for annotation in answers:
                if not isinstance(annotation, dict) or not isinstance(
                    annotation.get("answer"), dict
                ):
                    raise ExternalDatasetError("Invalid QASPER answer annotation.")
                references.append(
                    _qasper_reference(
                        annotation["answer"], text_evidence_only=text_evidence_only
                    )
                )
            cases.append(
                QasperCase(
                    paper_id=str(paper_id),
                    title=str(paper.get("title") or paper_id),
                    question_id=str(qa["question_id"]),
                    question=str(qa["question"]),
                    source_split=source_split,
                    references=references,
                )
            )
            if limit is not None and len(cases) >= limit:
                return cases
    return cases


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ExternalDatasetError(f"{path}:{line_number} is not a JSON object.")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalDatasetError(f"Could not load {path}: {exc}") from exc
    return rows


def load_scifact(
    corpus_path: str | Path,
    claims_path: str | Path,
    *,
    source_split: str,
    limit: int | None = None,
) -> list[SciFactCase]:
    """Read labeled native SciFact claims and resolve rationale sentence text."""
    corpus = {int(row["doc_id"]): row for row in _read_jsonl(Path(corpus_path))}
    cases = []
    for claim in _read_jsonl(Path(claims_path)):
        rationales = []
        labels = set()
        evidence = claim.get("evidence") or {}
        if not isinstance(evidence, dict):
            raise ExternalDatasetError(f"SciFact claim {claim.get('id')} has invalid evidence.")
        for raw_doc_id, evidence_sets in evidence.items():
            doc_id = int(raw_doc_id)
            document = corpus.get(doc_id)
            if document is None:
                raise ExternalDatasetError(f"SciFact evidence document {doc_id} is missing.")
            abstract = document.get("abstract")
            if not isinstance(abstract, list) or not all(isinstance(item, str) for item in abstract):
                raise ExternalDatasetError(f"SciFact document {doc_id} has invalid abstract.")
            for evidence_set in evidence_sets:
                label = evidence_set.get("label")
                indices = evidence_set.get("sentences")
                if label not in {"SUPPORT", "CONTRADICT"} or not isinstance(indices, list):
                    raise ExternalDatasetError(f"Invalid SciFact rationale for document {doc_id}.")
                try:
                    sentences = [abstract[int(index)] for index in indices]
                except (IndexError, TypeError, ValueError) as exc:
                    raise ExternalDatasetError(
                        f"SciFact rationale index is invalid for document {doc_id}."
                    ) from exc
                labels.add(label)
                rationales.append(
                    SciFactRationale(
                        doc_id=doc_id,
                        label=label,
                        sentence_indices=[int(index) for index in indices],
                        sentences=sentences,
                    )
                )
        if len(labels) > 1:
            raise ExternalDatasetError(
                f"SciFact claim {claim.get('id')} has conflicting gold labels."
            )
        case_label = next(iter(labels)) if labels else "NOT_ENOUGH_INFO"
        cited_doc_ids = [int(item) for item in claim.get("cited_doc_ids", [])]
        documents = []
        for doc_id in cited_doc_ids:
            document = corpus.get(doc_id)
            if document is None:
                raise ExternalDatasetError(f"SciFact cited document {doc_id} is missing.")
            abstract = document.get("abstract")
            if not isinstance(abstract, list) or not abstract or not all(
                isinstance(item, str) for item in abstract
            ):
                raise ExternalDatasetError(f"SciFact document {doc_id} has invalid abstract.")
            documents.append(
                SciFactDocument(
                    doc_id=doc_id,
                    title=str(document.get("title") or doc_id),
                    abstract=abstract,
                )
            )
        cases.append(
            SciFactCase(
                claim_id=int(claim["id"]),
                claim=str(claim["claim"]),
                source_split=source_split,
                label=case_label,
                cited_doc_ids=cited_doc_ids,
                documents=documents,
                rationales=rationales,
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases
