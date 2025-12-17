# Contributing to trueform

We welcome contributions—bug fixes, new features, documentation, examples, and benchmarks.

## Start with GitHub Issues

Browse [open issues](https://github.com/xlabmedical/trueform/issues) labeled by difficulty:

- **🟢 `difficulty: easy`** — Good first issues. Bug fixes, documentation, straightforward enhancements.
- **🟡 `difficulty: medium`** — Features, refactors, complex bugs. Requires familiarity with the codebase.
- **🔴 `difficulty: hard`** — Novel algorithms, architectural changes, deep geometric reasoning.

Pick an issue, comment that you're working on it, and get started.

**Interested in joining our team?** If you successfully tackle a hard-difficulty issue, we'd love to talk. Reach out to [info@polydera.com](mailto:info@polydera.com).

## Building from Source

```bash
git clone https://github.com/xlabmedical/trueform.git
cd trueform
mkdir build && cd build
cmake ..
make -j8
```

Build examples:
```bash
make examples -j8
```

## Workflow

1. **Fork the repo** and create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes.** See code guidelines below.

3. **Commit with clear messages:**
   ```bash
   git commit -m "Add mesh arrangements dual function"
   git commit -m "Fix intersection curve indexing bug"
   ```

4. **Push and open a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

   Reference the issue (e.g., "Fixes #42") and describe what you did.

## Code Guidelines

Trueform is built on composability and zero-copy semantics. When contributing:

- **Build on existing algorithms.** Implement new functionality using already-written parallel algorithms. Don't reinvent parallelism—compose it.
- **One function, one file.** Each user-facing function in `tf::` namespace gets its own file in the module directory.
- **Small functions, delegated responsibilities.** Break complex operations into small, focused functions. Each should do one thing clearly.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for template arguments.
- **Minimize comments.** Code should be self-explanatory. Comment only non-obvious algorithmic choices.

## Documentation

If you add a public API, update `docs/content/` with:
- What the function does (brief, one sentence)
- Example usage
- Return value description if non-obvious

Follow the existing structure in the modules documentation.

## Licensing

By contributing, you certify that:

1. Your work is provided under the **PolyForm Noncommercial License 1.0.0** for noncommercial use.
2. XLAB may offer your work under its **commercial license** without requiring additional paperwork from you.

If your employer owns your work, ensure they're comfortable with this arrangement.

## Questions?

- **Technical questions:** [GitHub Discussions](https://github.com/xlabmedical/trueform/discussions)
- **Bug reports:** [Open an issue](https://github.com/xlabmedical/trueform/issues)
- **Commercial licensing or jobs:** [info@polydera.com](mailto:info@polydera.com)

---

Thank you for contributing to trueform.
