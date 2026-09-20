"""The generator and distribution vocabulary shared by planning and generation.

Planning constrains model output to these names; generation resolves them to
Faker calls. Keeping both sides on one list stops a plan from naming a
generator that cannot be built.
"""

from __future__ import annotations

DEFAULT_GENERATOR = "auto"
DEFAULT_DISTRIBUTION = "uniform"

SUPPORTED_DISTRIBUTIONS = frozenset({"uniform", "sequential"})

TYPE_GENERATORS = frozenset(
    {
        "auto",
        "boolean",
        "date",
        "datetime",
        "decimal",
        "default",
        "enum",
        "float",
        "int",
        "integer",
        "number",
        "numeric",
        "random",
        "timestamp",
    }
)

GENERATOR_ALIASES = {
    "name": "full_name",
    "full_name": "full_name",
    "person_name": "full_name",
    "first_name": "first_name",
    "given_name": "first_name",
    "last_name": "last_name",
    "surname": "last_name",
    "family_name": "last_name",
    "email": "email",
    "email_address": "email",
    "phone": "phone",
    "phone_number": "phone",
    "telephone": "phone",
    "address": "address",
    "street_address": "address",
    "city": "city",
    "state": "state",
    "state_abbr": "state",
    "province": "state",
    "country": "country",
    "zip": "zip_code",
    "zip_code": "zip_code",
    "postcode": "zip_code",
    "postal_code": "zip_code",
    "company": "company",
    "company_name": "company",
    "employer": "company",
    "organization": "company",
    "url": "url",
    "website": "url",
    "uri": "url",
    "link": "url",
    "text": "text",
    "paragraph": "text",
    "description": "text",
    "sentence": "sentence",
    "title": "sentence",
    "word": "word",
    "job": "job",
    "job_title": "job",
    "isbn": "isbn",
    "label": "label",
    "license": "license",
    "uuid": "uuid",
    "time": "time",
    "sequence": "sequence",
}

GENERATOR_NAMES = tuple(sorted(set(GENERATOR_ALIASES) | TYPE_GENERATORS))


def normalize_hint(value: str | None) -> str:
    """Fold a generator or semantic-type hint to its lookup form."""
    if not value:
        return ""
    return value.strip().casefold().replace(" ", "_").replace("-", "_")


def is_supported_hint(value: str | None) -> bool:
    """Report whether a hint names a generator the data generator can build."""
    token = normalize_hint(value)
    return token in GENERATOR_ALIASES or token in TYPE_GENERATORS


def canonical_generator(value: str | None) -> str | None:
    """Resolve a hint to its Faker generator name.

    ## Returns
    The canonical name, or `None` when the hint names no semantic generator.
    """
    return GENERATOR_ALIASES.get(normalize_hint(value))
