# Question Answering Skill

(No learned rules yet. Rules will be added through the reflection process.)

## Rules

1. **Context overrides prior knowledge; do not substitute context terms with outside-knowledge equivalents.** The provided context may contain information that differs from real-world facts (e.g., counterfactual scenarios). If the context uses a specific term (e.g., "Somaliland"), use that exact term — do not replace it with a more commonly known alternative (e.g., "Somalia") even if you believe they refer to the same entity.

2. **Match the entity type the question asks for; answer the precise question asked.** If the question requests "universities," "companies," "people," or another specific entity type, provide answers in that form (e.g., "University of Colorado") rather than shortened or related forms (e.g., "Colorado"). Pay close attention to what specific information the question requests — give exactly what is asked, no more and no less.

3. **Strip possessive suffixes from brand/proper names unless integral.** When the answer is a brand or proper name, use the base form without possessive suffixes (e.g., "M&M" not "M&M's", "Ford" not "Ford's") unless the possessive is a core part of the name as established in the context.

4. **Match on shared descriptive details when terms are absent.** If the question uses a name or term that doesn't appear verbatim in the context, locate the answer by matching other descriptive attributes (dates, roles, locations, characteristics) that appear in both the question and context.

5. **Prefer exact wording from the context.** When the context contains a direct phrase that answers the question, copy it verbatim rather than paraphrasing or rewording.

6. **Keep answers minimal; do not over-specify.** Provide only the specific entity or short phrase requested — a single name or term, not a list. Do not include explanatory text, hedging, or restatements of the question. Do not append aliases, alternative names, qualifiers, or descriptive modifiers from the context unless the question explicitly asks for them (e.g., answer "Mice" not "house mice" if only the animal is asked for). When the question asks for one entity, provide exactly one — do not list multiple even if the context mentions several related ones.

7. **Do not prepend titles/honorifics or append trailing descriptive clauses.** When the answer is a person, use the name without titles like "King," "Queen," "President," or "Dr." (e.g., "Harold II" not "King Harold II") unless the title is an inseparable part of the name (e.g., "Alexander the Great"). When the context describes an entity with a relative clause (e.g., "broken-hearted people who cry about lost love"), extract only the core entity ("broken-hearted people") without the trailing description.

<!-- SLOW_UPDATE_START -->
1. **Preserve the exact numeral form from the context.** If the context writes "World War 1" with an Arabic numeral, do not convert it to "World War I" with a Roman numeral — and vice versa. Copy the numeral form exactly as it appears in the context. This applies to all numbers: dates, war names, quantities, etc.

2. **Preserve exact hyphenation and spelling from the context, even if it looks grammatically non-standard.** If the context writes "broken hearted" without a hyphen, do not add one. If it writes "well-known" with a hyphen, keep it. Never correct, standardize, or normalize the context's spelling or punctuation.

3. **When a question describes something as "acronymic" or references that it has an acronym, provide the full name — not the acronym itself.** Saying something is "acronymic" means it is known by an acronym; it does NOT mean the answer should be the acronym. If the context says "Open University (OU)," answer "Open University," not "OU."

4. **Include parenthetical qualifiers that appear within a name in the context.** When the context presents a name with a parenthetical component — e.g., "Tatiana (Szabova)" — include the full form with the parenthetical in your answer. This is different from a separate alias listed elsewhere; a parenthetical within the same name entry is part of the name itself. Rule 6's prohibition on appending aliases does not apply to parenthetical components that are part of a single name form in the context.
<!-- SLOW_UPDATE_END -->
