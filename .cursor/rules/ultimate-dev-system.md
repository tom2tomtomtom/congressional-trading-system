# Ultimate Dev System - Cursor Rules

## Specialist Activation
When Claude suggests using specialists, activate them based on:

### Frontend Work
- `frontend-developer` - UI components, styling, layouts
- `ui-engineer` - Component architecture, design systems
- `typescript-expert` - Type safety, complex TypeScript

### Backend Work  
- `backend-architect` - API design, server architecture
- `database-optimizer` - Database queries, performance
- `security-auditor` - Authentication, data protection

### DevOps & Deployment
- `deployment-engineer` - CI/CD, hosting setup
- `performance-engineer` - Optimization, monitoring

### Testing & Quality
- `test-automator` - Unit tests, integration tests
- `qa-engineer` - Manual testing, quality assurance

## Tool Routing Rules
- **0-5K lines**: Stay in Cursor for speed
- **5K-15K lines**: Cursor with careful context management  
- **15K+ lines**: Consider transitioning to Augment
- **Testing**: Use Claude Code for comprehensive test suites

## Context Management
- Always include project-specific context from CLAUDE.md
- Reference architecture decisions from previous sessions
- Maintain consistency with established patterns

## File Naming Conventions
- Use descriptive, kebab-case filenames
- Include component type in filename (Button.component.tsx)
- Group related files in feature directories
