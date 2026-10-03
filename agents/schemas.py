"""Pydantic schemas for P4 artifacts — validates coder output before review."""
from __future__ import annotations
from pydantic import BaseModel, field_validator, model_validator
import re


class SFCForwardingRule(BaseModel):
    """One entry in the sfc_forward P4 table."""
    spi: int          # Service Path Identifier (24-bit)
    si: int           # Service Index (8-bit)
    action: str       # "forward" | "drop"
    port: int | None = None  # required when action == "forward"

    @field_validator("spi")
    @classmethod
    def spi_range(cls, v):
        if not (0 <= v <= 0xFFFFFF):
            raise ValueError(f"SPI {v} out of 24-bit range")
        return v

    @field_validator("si")
    @classmethod
    def si_range(cls, v):
        if not (0 <= v <= 255):
            raise ValueError(f"SI {v} out of 8-bit range")
        return v

    @model_validator(mode="after")
    def port_required_for_forward(self):
        if self.action == "forward" and self.port is None:
            raise ValueError("port is required when action is 'forward'")
        return self


class SFCChain(BaseModel):
    """A full SFC chain definition."""
    spi: int
    name: str
    hops: list[SFCForwardingRule]

    @model_validator(mode="after")
    def si_decrements(self):
        sis = [h.si for h in self.hops]
        for i in range(1, len(sis)):
            if sis[i] >= sis[i - 1]:
                raise ValueError(
                    f"SI must decrement at each hop: hop {i} SI={sis[i]} >= hop {i-1} SI={sis[i-1]}"
                )
        return self


class P4CodeArtifact(BaseModel):
    """Validated P4 code output from the coder agent."""
    code: str
    target: str = "bmv2"

    @field_validator("code")
    @classmethod
    def has_p4_structure(cls, v):
        # Accept table-only output (control plane update) or full programs
        required = ["table", "action"]
        missing = [kw for kw in required if kw not in v]
        if missing:
            raise ValueError(f"P4 code missing required keywords: {missing}")
        return v

    @field_validator("code")
    @classmethod
    def no_unbounded_loops(cls, v):
        # Basic check: for loops without exit conditions are a red flag in P4
        if re.search(r'\bfor\s*\(', v):
            raise ValueError("P4_16 does not support for loops — use table actions instead")
        return v

    @property
    def line_count(self) -> int:
        return len(self.code.splitlines())


def validate_p4_code(code: str) -> tuple[bool, str]:
    """Validate P4 code and return (is_valid, error_message)."""
    try:
        P4CodeArtifact(code=code)
        return True, ""
    except Exception as e:
        return False, str(e)
