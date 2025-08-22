# Analyze Codebase Workflow

<workflow_meta>
  <name>analyze-codebase</name>
  <description>Deep analysis of Congressional Trading Intelligence System codebase</description>
  <estimated_time>10-15 minutes</estimated_time>
  <frequency>on-demand or when significant changes made</frequency>
</workflow_meta>

## Workflow Steps

<workflow_steps>
  <step number="1">
    <command>@structure-analysis</command>
    <description>Analyze project structure and organization</description>
    <condition>always</condition>
    <actions>
      - Review src/ directory organization
      - Check module interdependencies
      - Validate separation of concerns (core, analysis, dashboard)
      - Identify any structural issues or improvements
    </actions>
  </step>
  
  <step number="2">
    <command>@code-quality-check</command>
    <description>Evaluate code quality and standards</description>
    <condition>always</condition>
    <tools>black, flake8, mypy (if configured)</tools>
    <actions>
      - Check Python PEP8 compliance
      - Analyze function complexity and organization
      - Review import statements and dependencies
      - Identify potential refactoring opportunities
    </actions>
  </step>
  
  <step number="3">
    <command>@dependency-review</command>
    <description>Review dependencies and security</description>
    <condition>always</condition>
    <actions>
      - Analyze requirements.txt and pyproject.toml
      - Check for outdated or vulnerable packages
      - Review optional dependencies usage
      - Validate production vs development dependencies
    </actions>
  </step>
  
  <step number="4">
    <command>@intelligence-engine-analysis</command>
    <description>Analyze core intelligence and analysis modules</description>
    <condition>always</condition>
    <focus_areas>
      - src/core/intelligence_fusion_engine.py
      - src/analysis/congressional_analysis.py
      - src/analysis/enhanced_analysis.py
      - ML model implementations and data flow
    </focus_areas>
  </step>
  
  <step number="5">
    <command>@dashboard-analysis</command>
    <description>Review dashboard implementation and integration</description>
    <condition>always</condition>
    <focus_areas>
      - HTML/CSS/JS dashboard functionality
      - React components (if present)
      - API integration points
      - User experience and accessibility
    </focus_areas>
  </step>
  
  <step number="6">
    <command>@data-pipeline-review</command>
    <description>Analyze data collection and processing pipeline</description>
    <condition>always</condition>
    <focus_areas>
      - Data sources and API integrations
      - Data validation and quality checks
      - Sample data vs production data strategy
      - Performance and scalability considerations
    </focus_areas>
  </step>
</workflow_steps>

## Context Awareness

**Branch-Specific Analysis**
- `main`: Focus on stability, documentation, and production readiness
- `feature/data-expansion`: Emphasize data integrity, API integration, and scalability
- `feature/*`: Development-focused analysis with attention to integration points

**Component-Specific Analysis**
- **Intelligence Engine Changes**: Deep dive into algorithm accuracy and performance
- **Dashboard Updates**: UI/UX analysis, responsiveness, and data visualization
- **Data Model Changes**: Validation of schema integrity and migration paths
- **API Integration**: Security, rate limiting, and error handling analysis

## Output Format

### Summary Report
- **Overall Health Score**: 1-10 rating with justification
- **Critical Issues**: High-priority problems requiring immediate attention
- **Technical Debt**: Areas needing refactoring or improvement
- **Strengths**: Well-implemented components and patterns

### Detailed Analysis
- **Architecture Review**: Adherence to modular design principles
- **Code Quality**: Specific recommendations for improvement
- **Performance**: Bottlenecks and optimization opportunities
- **Security**: Potential vulnerabilities and hardening recommendations

### Action Items
- **Immediate**: Issues requiring urgent fixes
- **Short-term**: Improvements to implement in next development cycle
- **Long-term**: Strategic architectural enhancements

## Success Criteria

- [ ] No critical security vulnerabilities detected
- [ ] Code quality metrics within acceptable ranges
- [ ] Modular architecture properly maintained
- [ ] Dependencies up-to-date and secure
- [ ] Intelligence engine algorithms functioning correctly
- [ ] Dashboard responsive and functional
- [ ] Data pipeline processing efficiently
- [ ] Documentation accurate and up-to-date

## Integration Points

**Pre-Development**: Run before starting significant feature work
**Post-Development**: Run after completing major changes
**Pre-Deployment**: Part of deployment validation checklist
**Scheduled**: Weekly comprehensive analysis during active development