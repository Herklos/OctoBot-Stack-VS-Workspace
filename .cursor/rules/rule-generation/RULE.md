---
description: "Guidelines for creating, formatting, and maintaining Cursor rules. Use when creating or updating rule files."
globs: [".cursor/rules/**/RULE.md"]
alwaysApply: false
---

# Cursor Rules Generation Guidelines

## Rule File Structure

### Frontmatter (Required)

Every rule file must start with YAML frontmatter:

```yaml
---
description: "Clear, concise description of what this rule covers"
alwaysApply: false  # or true
globs: []  # Optional: file patterns for "Apply to Specific Files"
---
```

**Frontmatter fields:**
- `description`: Required for "Apply Intelligently" rules. Should be clear and searchable.
- `alwaysApply`: `true` for rules that should apply to every chat, `false` for intelligent/manual application
- `globs`: Array of file patterns (e.g., `["**/*.py", "**/tests/**"]`) for file-specific rules

### Rule Types

Choose the appropriate rule type based on use case:

1. **Always Apply** (`alwaysApply: true`)
   - Use for fundamental, always-relevant rules
   - Applied to every chat session
   - Keep these minimal and essential

2. **Apply Intelligently** (`alwaysApply: false` with `description`)
   - Use for context-specific rules
   - Agent decides relevance based on description
   - Best for most rules

3. **Apply to Specific Files** (`globs` with patterns)
   - Use when rule only applies to certain file types
   - Example: `globs: ["**/*.tsx", "**/components/**"]`

4. **Apply Manually** (`alwaysApply: false`, no globs)
   - Use for specialized rules invoked with `@rule-name`
   - Good for templates or rarely-used workflows

## Content Formatting

### Use Action-Oriented Language

**Good:**
```
When creating exchange tentacles:
- Inherit from `exchanges.RestExchange`
- Add `DESCRIPTION = ""` class variable
```

**Bad:**
```
Exchange tentacles have:
- Base class: RestExchange
- Description attribute exists
```

### Structure with Context

Start sections with "When..." or "In..." to provide context:

```
When working with tentacles:
- Place files in correct directory structure
- Include required metadata.json

In test files:
- Use pytest with async markers
- Import test utilities from octobot_commons.tests
```

### Keep Rules Focused

- **One rule file = One concern** (e.g., imports, testing, tentacles)
- Split large rules into multiple focused rules
- Keep rules under 500 lines (preferably under 200)

### Use Code Examples

Include concrete examples:

```python
# Good example
import octobot_trading.exchanges as exchanges

class MyExchange(exchanges.RestExchange):
    DESCRIPTION = ""
    DEFAULT_CONNECTOR_CLASS = MyConnector
    
    @classmethod
    def get_name(cls):
        return "myexchange"
```

### Reference Files with @ Syntax

Reference example files in your rules:

```
Use this template when creating Express services:

@express-service-template.ts
```

## Best Practices

### Clarity and Precision

- **Be specific**: "Use `pytest.mark.asyncio`" not "Use async testing"
- **Provide examples**: Show code patterns, not just describe them
- **Use correct terminology**: Match the codebase's actual naming

### Scoping

- **Scope by context**: "When creating tentacles..." not "Always..."
- **Scope by file type**: Use `globs` for file-specific rules
- **Scope by directory**: Reference directory patterns when relevant

### Maintainability

- **Update rules when codebase changes**: Rules should reflect current patterns
- **Remove outdated rules**: Don't keep deprecated patterns
- **Version appropriately**: Update descriptions when rules change significantly

### Organization

- **Group related rules**: Use clear section headers
- **Order by importance**: Most critical rules first
- **Use consistent formatting**: Same structure across similar rules

## Rule File Naming

- Use descriptive, kebab-case names: `exchange-tentacles.md`, `testing-conventions.md`
- Group related rules in subdirectories if needed
- Avoid generic names like `rules.md` or `general.md` (unless truly general)

## Examples of Good Rules

### Example 1: Focused, Action-Oriented

```markdown
---
description: "Standards for creating exchange tentacles in OctoBot"
alwaysApply: false
---

When creating exchange tentacles:

Inherit from `exchanges.RestExchange`:
```python
class MyExchange(exchanges.RestExchange):
    DESCRIPTION = ""
    DEFAULT_CONNECTOR_CLASS = MyConnector
    
    @classmethod
    def get_name(cls):
        return "myexchange"
```

File naming: `{exchange_name}_exchange.py`

@polymarket_exchange.py
```

### Example 2: Context-Specific with Globs

```markdown
---
description: "Python import conventions for OctoBot repositories"
globs: ["**/*.py"]
alwaysApply: false
---

When importing in Python files:

- Use absolute imports with aliases: `import octobot_trading.exchanges as exchanges`
- Follow pattern: `import octobot_{repo}.{module} as {alias}`
- No circular dependencies
```

## Common Mistakes to Avoid

1. **Too vague**: "Follow best practices" → Be specific
2. **Too long**: 1000+ line rules → Split into focused rules
3. **Missing description**: Required for intelligent application
4. **Outdated examples**: Keep code examples current
5. **Conflicting rules**: Ensure rules don't contradict each other
6. **No context**: "Do X" → "When Y, do X"

## Updating Rules

When updating existing rules:

1. **Check current codebase patterns**: Verify rules match actual code
2. **Update examples**: Ensure code examples are current
3. **Revise description**: Update if scope changes
4. **Test applicability**: Ensure rule triggers appropriately
5. **Document changes**: Note significant changes in rule content

## Rule Validation Checklist

Before finalizing a rule, verify:

- [ ] Frontmatter is correctly formatted
- [ ] Description is clear and searchable
- [ ] Rule type (alwaysApply/globs) is appropriate
- [ ] Content is action-oriented ("When...", "In...")
- [ ] Examples are current and accurate
- [ ] Rule is focused (single concern)
- [ ] Under 500 lines (preferably under 200)
- [ ] File naming follows conventions
- [ ] No conflicts with existing rules

