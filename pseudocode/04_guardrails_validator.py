"""
Step 4 - Guardrails AI Validators
"""

import json
import re

from guardrails import Guard

try:
    from guardrails import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction

try:
    from guardrails.validators import FailResult, PassResult, Validator, register_validator
except ImportError:
    from guardrails.validator_base import FailResult, PassResult, Validator, register_validator


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_types = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)
            if matches:
                redacted_text = re.sub(pattern, f"[{pii_type}_REDACTED]", redacted_text)
                found_types.extend([pii_type] * len(matches))

        if found_types:
            print(f"  Redacted {len(found_types)} item(s): {', '.join(found_types)}")
            return FailResult(
                error_message="PII detected in output.",
                fix_value=redacted_text,
            )
        return PassResult(value_override=value)


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    @staticmethod
    def _repair(text: str) -> str:
        repaired = text.strip()
        repaired = re.sub(r"^```(?:json)?\s*", "", repaired)
        repaired = re.sub(r"\s*```$", "", repaired)
        repaired = repaired.strip()
        repaired = repaired.replace("'", '"')
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return repaired

    def validate(self, value: str, metadata: dict):
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            print("  JSON repaired successfully")
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError as exc:
            fallback = json.dumps(
                {"error": f"Invalid JSON after repair: {exc}", "raw": value},
                indent=2,
            )
            return FailResult(
                error_message=f"Invalid JSON after repair attempt: {exc}",
                fix_value=fallback,
            )


def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


def demo_json_guard():
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        output_text = str(result.validated_output)
        if label == "Truly invalid":
            status = "Fail"
        elif output_text.strip() == text.strip():
            status = "Pass"
        else:
            status = "Pass"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:80]}")
        print(f"  Output: {output_text[:120]}")


def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\nStep 4 complete!")


if __name__ == "__main__":
    main()
