# Daily Development Workflow

<workflow_meta>
  <name>daily-dev</name>
  <description>Standard daily development workflow for Congressional Trading Intelligence System</description>
  <estimated_time>15-30 minutes</estimated_time>
  <frequency>daily</frequency>
</workflow_meta>

## Workflow Steps

<workflow_steps>
  <step number="1">
    <command>@analyze-codebase</command>
    <description>Review current codebase state and identify issues</description>
    <condition>always</condition>
    <expected_output>Code quality report, technical debt assessment</expected_output>
  </step>
  
  <step number="2">
    <command>@test-intelligence-engine</command>
    <description>Run congressional analysis engine tests</description>
    <condition>if_analysis_changes</condition>
    <script>python3 src/analysis/congressional_analysis.py</script>
  </step>
  
  <step number="3">
    <command>@update-dashboard-data</command>
    <description>Refresh dashboard with latest sample data</description>
    <condition>if_data_changes</condition>
    <action>Check dashboard loads correctly with updated data</action>
  </step>
  
  <step number="4">
    <command>@clean-codebase</command>
    <description>Apply code formatting and linting</description>
    <condition>if_code_issues_found</condition>
    <tools>black, flake8</tools>
  </step>
  
  <step number="5">
    <command>@check-dependencies</command>
    <description>Verify all required packages are installed</description>
    <condition>if_requirements_changed</condition>
    <script>pip install -r requirements.txt</script>
  </step>
  
  <step number="6">
    <command>@pre-deploy-check</command>
    <description>Run pre-deployment validation</description>
    <condition>if_significant_changes</condition>
    <reference>@.claude-suite/project/checklists/pre-deploy.md</reference>
  </step>
</workflow_steps>

## Context Awareness

**Branch Detection**
- `main`: Focus on stability and documentation
- `feature/data-expansion`: Emphasize data integrity and API integration
- `feature/*`: Development-focused with extensive testing

**Component Focus**
- **Intelligence Engine Changes**: Extra testing of analysis algorithms
- **Dashboard Updates**: Visual regression testing and responsiveness
- **Data Model Changes**: Validation of data integrity and schema

## Success Criteria

- [ ] All tests pass
- [ ] Code quality metrics maintained
- [ ] No regressions in core functionality
- [ ] Dashboard loads and displays correctly
- [ ] Dependencies up to date and secure

## Emergency Procedures

**If Critical Issues Found:**
1. Stop current work
2. Assess impact and severity
3. Create hotfix branch if on main
4. Document issue in `.claude-suite/project/issues.md`
5. Implement fix and test thoroughly

**If Data Corruption Detected:**
1. Backup current state
2. Identify corruption source
3. Restore from known good state
4. Implement additional validation

## Metrics Tracking

**Daily Metrics:**
- Code coverage percentage
- Number of failing tests
- Dashboard load time
- Memory usage of analysis engine
- Number of congressional members tracked

**Weekly Review:**
- Technical debt accumulation
- Feature completion progress
- Performance benchmark trends
- User feedback and issues