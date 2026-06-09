from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_REF_MARKERS = {"ref", "reference"}


def _normalize_type_token(token: str) -> str:
    """Normalize the leading type token to a bare subtype string.

    Hyphenated forms involving 'ref' or 'reference' collapse to the subtype:
    ``ref-air`` / ``air-ref`` / ``reference-air`` / ``air-reference`` → ``air``.
    A plain ``reference`` with no subtype stays as ``reference``. Tokens that
    contain hyphens but no ref-marker (e.g. ``two-photon``) are preserved as-is.
    """
    lower = token.lower()
    if "-" not in lower:
        return lower
    parts = lower.split("-")
    non_ref = [p for p in parts if p not in _REF_MARKERS]
    if len(non_ref) == len(parts):
        return lower
    if not non_ref:
        return "reference"
    if len(non_ref) == 1:
        return non_ref[0]
    return "-".join(non_ref)


@dataclass
class FilenameInfo:
    """Parsed metadata derived from a filename.

    Filename grammar (controlled by ``delimiter``, default ``_``):

    ``<type>[_<token>]*.<ext>``

    where each ``<token>`` is either positional (matched by index against the
    user-provided ``keywords`` list) or of the form ``key=value`` (extracted as
    an attribute regardless of position). The leading ``<type>`` token is
    normalised via :func:`_normalize_type_token` so ``ref-air``, ``air-ref``,
    ``reference-air`` and ``air-reference`` all collapse to ``data_type='air'``.

    Examples:
        ``sample_100K_up.dat``                    — positional
        ``sample_temp=100K_series=up.dat``        — explicit
        ``sample_100K_aperture=open.dat``         — mixed
        ``ref-substrate_temp=100K_series=up.dat`` — substrate reference

    This object is intentionally lightweight and decoupled from service logic.
    Matching/pairing behaviour remains in :class:`GroupingService`.
    """

    filename: str
    keywords: list[str] = field(default_factory=lambda: ["type", "series", "temp"])
    report_list: list[str] = field(default_factory=lambda: ["data_type", "series"])

    series: str | None = None
    data_type: str | None = None
    temperature: str | None = None

    extra_details: list[str] | None = None
    air_reference: str | None = None
    substrate_reference: str | None = None

    @classmethod
    def from_filename(
        cls,
        filename: str,
        *,
        delimiter: str = "_",
        keywords: list[str] | None = None,
        merge_extra: bool = False,
        **kwargs: Any,
    ) -> "FilenameInfo":
        selected_keywords = keywords if keywords is not None else ["type", "series", "temp"]
        item = cls(filename=filename, keywords=list(selected_keywords))
        item.parse(
            delimiter=delimiter,
            keywords=item.keywords,
            merge_extra=merge_extra,
            **kwargs,
        )
        return item

    def parse(
        self,
        *,
        delimiter: str = "_",
        keywords: list[str] | None = None,
        merge_extra: bool = False,
        **kwargs: Any,
    ) -> None:
        """Parse this filename into fields.

        Tokens containing ``=`` are extracted first into named attributes
        regardless of position. The remaining bare tokens are matched against
        ``keywords`` by index. The first bare token is treated as the ``type``
        and run through :func:`_normalize_type_token`.

        Positional tokens beyond the ``keywords`` length go into
        ``extra_details``; when ``merge_extra=True`` they are also appended to
        ``series`` so the existing pairing logic can disambiguate.
        """

        active_keywords = keywords if keywords is not None else self.keywords
        self.keywords = list(active_keywords)

        self.report_list = ["data_type", "series"]
        self.series = None
        self.data_type = None
        self.temperature = None
        self.extra_details = None

        stem = ".".join(self.filename.split(".")[:-1]) or self.filename
        raw_components = [c.strip() for c in stem.split(delimiter) if c.strip()]

        # Step 1: pull out key=value tokens regardless of position.
        positional: list[str] = []
        for comp in raw_components:
            if "=" not in comp:
                positional.append(comp)
                continue
            key, _, value = comp.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                # malformed (=value with empty key) — treat as positional
                positional.append(comp)
                continue
            self.__dict__[key] = value
            if key not in self.report_list:
                self.report_list.append(key)

        # Step 2: match remaining positional tokens against the keyword list.
        for index, key in enumerate(self.keywords):
            if index >= len(positional):
                break

            value = positional[index]
            if key == "type":
                self.data_type = _normalize_type_token(value)
            elif key == "series":
                if "." in value:
                    indices = [i for i, ch in enumerate(value) if ch == "."]
                    value = value[: indices[-1]]
                self.series = value
            else:
                self.__dict__[key] = value
                if key not in self.report_list:
                    self.report_list.append(key)

        # Step 3: anything past the keyword list becomes extras (positional only).
        if len(positional) > len(self.keywords):
            extras = positional[len(self.keywords) :]
            if merge_extra:
                self.series = delimiter.join([self.series, *extras]) if self.series else delimiter.join(extras)
            self.extra_details = extras

    def __repr__(self) -> str:
        info_lines = []
        for key in self.report_list:
            info_lines.append(f" -> {key}: {self.__dict__.get(key, None)}")

        for key in ["extra_details", "air_reference", "substrate_reference"]:
            info_lines.append(f" -> {key}: {self.__dict__.get(key, None)}")

        return f"Filename: {self.filename}\n" + "\n".join(info_lines)

    def __str__(self) -> str:
        return self.filename
