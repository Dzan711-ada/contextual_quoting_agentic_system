"""Validated tool arguments and structured model outputs."""

from pydantic import BaseModel, ConfigDict, Field


class BusinessInformation(BaseModel):
    """A partial update. Omitted fields never erase information already collected."""

    legal_name: str | None = Field(None, description="Business legal name")
    primary_products_or_services: str | None = Field(None, description="Primary operations")
    secondary_or_ancillary_operations: str | None = Field(None, description="Other operations; 'None' if absent")
    industries: str | None = Field(None, description="Industries served")
    manufacturing_retail_wholesale_or_distribution: str | None = Field(None, description="Actual operating model, including software/services")
    projected_revenue: float | None = Field(None, gt=0, allow_inf_nan=False, description="Annual revenue in USD as a number; $4 million = 4000000")


class ToUnderwritingAssistant(BaseModel):
    """Begin classification analysis after collecting and summarizing business facts."""

    request: str


class FinalClassificationsInfo(BaseModel):
    """A confirmed classification copied from the retrieved catalog."""

    category: str
    classification_description: str
    code: int = Field(gt=0)


class ToQuoteAssistant(BaseModel):
    """Generate a quote only after the user has confirmed the classifications."""

    request: str
    final_classifications: list[FinalClassificationsInfo] = Field(min_length=1)


class CompleteOrEscalate(BaseModel):
    """Finish this task or return to the information assistant for corrections."""

    cancel: bool = True
    reason: str


class CategoryRationale(BaseModel):
    """A catalog classification with the model's justification."""

    category: str
    cat_description: str
    code: int = Field(gt=0)
    rate: float = Field(gt=0, allow_inf_nan=False)
    rationalization: str


class CategoriesOutput(BaseModel):
    """Classification candidates selected exclusively from retrieved records."""

    categories: list[CategoryRationale] = Field(min_length=1)


class DescriptionGrade(BaseModel):
    """Confidence is an internal 0–1 score, not an actuarial assessment."""

    category: str
    cat_description: str
    code: int = Field(gt=0)
    grade: float = Field(ge=0, le=1, allow_inf_nan=False)


class DescriptionGradeSchema(BaseModel):
    """Graded candidates returned by the grading model."""

    description_grades: list[DescriptionGrade] = Field(min_length=1)


class RateRequest(BaseModel):
    """Validated deterministic quote inputs."""

    model_config = ConfigDict(extra="forbid")
    categories: list[str] = Field(min_length=1)
    projected_revenue: float = Field(gt=0, allow_inf_nan=False)
