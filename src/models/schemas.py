"""Structured handoff schemas enforced for machine-readable agent outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiEndpoint(BaseModel):
    method: str = Field(min_length=1)
    path: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class DataEntity(BaseModel):
    name: str = Field(min_length=1)
    fields: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)


class ArchitectHandoffSchema(BaseModel):
    system_diagram_json: dict[str, Any]
    database_schema: list[DataEntity]
    api_endpoints: list[ApiEndpoint]
    design_decisions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    handoff_notes: str = Field(min_length=1)


class QuorumJudgeSchema(BaseModel):
    selected_option: Literal["A", "B", "merged"]
    rationale: str = Field(min_length=1)
    merged_architecture: ArchitectHandoffSchema


class GeneratedFileSchema(BaseModel):
    path: str = Field(min_length=1)
    content: str


class StandardAgentHandoffSchema(BaseModel):
    files: list[GeneratedFileSchema] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    status: Literal["success", "failure"] = "success"
    summary: str = ""
    handoff_notes: str = ""
