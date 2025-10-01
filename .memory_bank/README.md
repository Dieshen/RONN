# RONN Memory Bank

This directory contains the project memory bank for the RONN (Rust Open Neural Network Runtime) project. These files serve as persistent context for AI assistants working on the codebase.

## Files

### project_overview.md
High-level project identity, metrics, value propositions, and performance targets.

### crate_structure.md
Detailed breakdown of all workspace crates, their purposes, key components, and dependencies.

### implementation_status.md
Current implementation status, completed work, and prioritized roadmap for future development.

### critical_files.md
Quick reference for important file locations across the codebase with line number references.

### technical_decisions.md
Design rationale and tradeoff analysis for major technical decisions in the project.

## Usage Guidelines

### When to Read
- At the start of each session working on RONN
- When context is needed about project architecture
- Before implementing new features
- When making architectural decisions

### When to Update
- After completing major features or milestones
- When architectural decisions are made
- When new critical files are created
- When priorities change in the roadmap

### Update Frequency
- **After major changes**: Immediately update relevant files
- **During feature development**: Update implementation_status.md with progress
- **New architectural decisions**: Add to technical_decisions.md
- **New critical files**: Add to critical_files.md

## Memory Bank Philosophy

The memory bank follows these principles:

1. **Source of Truth**: Derive from code, commits, and documentation - never invent
2. **Actionable Context**: Focus on information needed for development decisions
3. **Living Documents**: Update regularly, keep synchronized with codebase
4. **Concise but Complete**: Balance detail with readability
5. **Cross-Referenced**: Link to actual code locations with line numbers

## Related Documentation

- `../README.md` - Project README with quick start
- `../TASKS.md` - Comprehensive development roadmap
- `../docs/` - Design documents and architecture guides
- `../.github/workflows/` - CI/CD configuration
- `../examples/` - Working code examples

## Maintenance

The memory bank should be reviewed and updated:
- Weekly during active development
- After completing tasks from TASKS.md
- When new crates are added
- When major refactoring occurs
- Before creating releases

Last Updated: 2025-10-01
Project Status: Post-MVP, production hardening phase
Code Size: 23,230+ lines
Test Coverage: ~7 basic tests (target: 80%+)
