# AGENTS.md - Agent Behavior Configuration

## Default Behavior
- Always ask before creating new files
- Prefer refactoring over rewriting
- Explain complex logic with comments
- Suggest tests for new features

## Code Review Checklist
Before marking task complete:
- [ ] Code follows project conventions
- [ ] No console.log statements
- [ ] Error handling implemented
- [ ] TypeScript types are correct
- [ ] Tests pass (if applicable)

## Communication Style
- Be concise but thorough
- Provide examples when explaining concepts
- Ask clarifying questions when requirements are unclear
- Suggest improvements, don't just follow orders

## Security Reminders
- Never commit .env files
- Sanitize all user inputs
- Use parameterized queries
- Validate JWT tokens on every request