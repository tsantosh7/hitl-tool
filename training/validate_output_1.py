import json
from jsonschema import validate, ValidationError

FIELDS = [
    "ConvCourtName",
    "ConvictPleaDate",
    "ConvictOffence",
    "AcquitOffence",
    "ConfessPleadGuilty",
    "PleaPoint",
    "RemandDecision",
    "RemandCustodyTime",
    "SentCourtName",
    "Sentence",
    "SentServe",
    "WhatAncillary",
    "OffSex",
    "OffAgeOffence",
    "OffJobOffence",
    "OffHomeOffence",
    "OffMentalOffence",
    "OffIntoxOffence",
    "OffVicRelation",
    "VictimType",
    "VicNum",
    "VicSex",
    "VicAgeOffence",
    "VicJobOffence",
    "VicHomeOffence",
    "VicMentalOffence",
    "VicIntoxOffence",
    "ProsEvidTypeTrial",
    "DefEvidTypeTrial",
    "PreSentReport",
    "AggFactSent",
    "MitFactSent",
    "VicImpactStatement",
    "Appellant",
    "CoDefAccNum",
    "AppealAgainst",
    "AppealGround",
    "SentGuideWhich",
    "AppealOutcome",
    "ReasonQuashConv",
    "ReasonSentExcessNotLenient",
    "ReasonSentLenientNotExcess",
    "ReasonDismiss",
]


def build_schema(fields):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": fields,
        "properties": {
            f: {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
            for f in fields
        },
    }


SCHEMA = build_schema(FIELDS)


def validate_output(output: dict):
    """
    Raises ValidationError if invalid.
    """
    validate(instance=output, schema=SCHEMA)


if __name__ == "__main__":
    # quick manual test
    example = {f: ["data not available"] for f in FIELDS}
    validate_output(example)
    print("Schema OK")
