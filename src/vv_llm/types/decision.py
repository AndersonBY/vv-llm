"""Provider-neutral decision requests and probability-bearing responses."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

DecisionType = Literal["predicate", "choice", "score"]
ChoiceValue = StrictStr | StrictBool
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
ScoreValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]
TokenCount = Annotated[StrictInt, Field(ge=0)]


class _RequestObject(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionText(_RequestObject):
    type: Literal["text"] = "text"
    text: StrictStr


class DecisionImage(_RequestObject):
    type: Literal["image_url"] = "image_url"
    image_url: Annotated[str, Field(strict=True, pattern=r"^data:image/[^;,]+;base64,[A-Za-z0-9+/]+={0,2}$")]
    detail: Literal["auto", "low", "high", "original"] | None = None


DecisionContent = Annotated[DecisionText | DecisionImage, Field(discriminator="type")]


class DecisionMessage(_RequestObject):
    type: Literal["message"] | None = None
    role: Literal["user"] = "user"
    content: StrictStr | Annotated[list[DecisionContent], Field(min_length=1)]


class _Question(_RequestObject):
    instructions: Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]
    name: StrictStr | None = None


class PredicateQuestion(_Question):
    type: Literal["predicate"] = "predicate"


class DecisionChoice(_RequestObject):
    id: ChoiceValue
    description: StrictStr


class ChoiceQuestion(_Question):
    type: Literal["choice"] = "choice"
    choices: Annotated[list[ChoiceValue | DecisionChoice], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_choices(self) -> ChoiceQuestion:
        ids = [choice.id if isinstance(choice, DecisionChoice) else choice for choice in self.choices]
        if len(set(ids)) != len(ids):
            raise ValueError("choice IDs must be unique")
        return self


class DecisionLevel(_RequestObject):
    label: StrictStr
    description: StrictStr | None = None


class ScoreQuestion(_Question):
    type: Literal["score"] = "score"
    rubric: StrictStr | Annotated[list[DecisionLevel], Field(min_length=1)]


DecisionQuestion = Annotated[PredicateQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(_RequestObject):
    model: Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")] = ""
    input: StrictStr | Annotated[list[DecisionMessage], Field(min_length=1)]
    questions: Annotated[list[DecisionQuestion], Field(min_length=1)]
    safety_identifier: Annotated[str, Field(strict=True, max_length=128)] | None = None

    @model_validator(mode="after")
    def validate_questions(self) -> DecisionRequest:
        names = [question.name for question in self.questions if question.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("question names must be unique")
        if isinstance(self.input, list):
            images = sum(isinstance(part, DecisionImage) for message in self.input if isinstance(message.content, list) for part in message.content)
            if images > 128:
                raise ValueError("decision input supports at most 128 images")
        return self

    @classmethod
    def from_contract(cls, value: dict[str, Any]) -> DecisionRequest:
        if "model" not in value:
            raise ValueError("canonical decision requests require model")
        for question in value.get("questions", []):
            if not isinstance(question, dict) or "type" not in question:
                raise ValueError("canonical questions require type")
        if isinstance(value.get("input"), list):
            for message in value["input"]:
                if not isinstance(message, dict) or "role" not in message:
                    raise ValueError("canonical messages require role")
                if isinstance(message.get("content"), list) and any(not isinstance(part, dict) or "type" not in part for part in message["content"]):
                    raise ValueError("canonical content requires type")
        return cls.model_validate(value)

    def to_contract(self) -> dict[str, Any]:
        value = self.model_dump(mode="json", exclude_none=True)
        self.from_contract(value)
        return value


class _Answer(BaseModel):
    name: StrictStr | None = None


class PredicateAnswer(_Answer):
    type: Literal["predicate"]
    probability: Probability


class ChoiceProbability(BaseModel):
    choice: ChoiceValue
    probability: Probability


class ChoiceAnswer(_Answer):
    type: Literal["choice"]
    choice: ChoiceValue
    options: Annotated[list[ChoiceProbability], Field(min_length=1)]


class LevelProbability(BaseModel):
    value: TokenCount
    label: StrictStr
    probability: Probability


class ScoreAnswer(_Answer):
    type: Literal["score"]
    score: ScoreValue
    confidence: Probability
    probabilities: Annotated[list[LevelProbability], Field(min_length=1)]


class RefusalAnswer(_Answer):
    type: Literal["refusal"]


DecisionAnswer = Annotated[PredicateAnswer | ChoiceAnswer | ScoreAnswer | RefusalAnswer, Field(discriminator="type")]


class DecisionInputTokenDetails(BaseModel):
    model_config = ConfigDict(extra="allow")
    cached_tokens: TokenCount | None = None
    cache_write_tokens: TokenCount | None = None


class DecisionOutputTokenDetails(BaseModel):
    model_config = ConfigDict(extra="allow")
    reasoning_tokens: TokenCount | None = None


class DecisionUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    total_tokens: TokenCount | None = None
    input_tokens_details: DecisionInputTokenDetails | None = None
    output_tokens_details: DecisionOutputTokenDetails | None = None


class DecisionResponse(BaseModel):
    model: Annotated[str, Field(strict=True, min_length=1)]
    answers: Annotated[list[DecisionAnswer], Field(min_length=1)]
    usage: DecisionUsage | None = None

    def to_contract(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)

    @classmethod
    def from_contract(cls, value: dict[str, Any]) -> DecisionResponse:
        response = cls.model_validate(value)
        if response.to_contract() != value:
            raise ValueError("unknown or missing canonical decision response fields")
        return response

    def validate_for(self, request: DecisionRequest) -> None:
        if len(self.answers) != len(request.questions):
            raise ValueError("decision answer count does not match questions")
        for question, answer in zip(request.questions, self.answers, strict=True):
            if answer.name != question.name or (answer.type != "refusal" and answer.type != question.type):
                raise ValueError("decision answer type/name does not match question")
            if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
                ids = [choice.id if isinstance(choice, DecisionChoice) else choice for choice in question.choices]
                options = [option.choice for option in answer.options]
                if len(options) != len(set(options)) or set(options) != set(ids) or answer.choice not in ids:
                    raise ValueError("decision choice does not match supplied choices")
            if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer) and isinstance(question.rubric, list):
                values = [level.value for level in answer.probabilities]
                if len(values) != len(set(values)) or set(values) != set(range(len(question.rubric))) or not 0 <= answer.score <= len(question.rubric) - 1:
                    raise ValueError("decision score does not match supplied rubric")
                if any(level.label != question.rubric[level.value].label for level in answer.probabilities):
                    raise ValueError("decision score labels do not match supplied rubric")
