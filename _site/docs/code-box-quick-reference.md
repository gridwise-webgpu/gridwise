# Quick Reference: Simple Code Box

## Copy-Paste Templates

### Template 1: Basic Code Box
```html
<div class="code-box-simple">
  <div class="code-box-header">Title Here</div>
  <div class="code-box-content">
    <pre><code>Your code here</code></pre>
  </div>
</div>
```

### Template 2: JavaScript/TypeScript
```html
<div class="code-box-simple">
  <div class="code-box-header">JavaScript Example</div>
  <div class="code-box-content">
    <pre><code><span class="keyword">const</span> myVar = <span class="string">'value'</span>
<span class="keyword">function</span> <span class="function">myFunction</span>(<span class="property">param</span>) {
  <span class="comment">// Your code</span>
  <span class="keyword">return</span> <span class="property">param</span> * <span class="number">2</span>
}</code></pre>
  </div>
</div>
```

### Template 3: WGSL
```html
<div class="code-box-simple">
  <div class="code-box-header">WGSL Shader</div>
  <div class="code-box-content">
    <pre><code><span class="keyword">var</span>&lt;<span class="builtin">workgroup</span>&gt; <span class="property">data</span>: <span class="type">array</span>&lt;<span class="type">u32</span>, <span class="number">256</span>&gt;;

<span class="keyword">fn</span> <span class="function">myFunction</span>(<span class="property">x</span>: <span class="type">u32</span>) -&gt; <span class="type">u32</span> {
  <span class="keyword">return</span> <span class="property">x</span> * <span class="number">2</span>;
}</code></pre>
  </div>
</div>
```

### Template 4: No Header
```html
<div class="code-box-simple">
  <div class="code-box-content">
    <pre><code>Code without header</code></pre>
  </div>
</div>
```

## Syntax Highlighting Classes

| Class | Color | Use For |
|-------|-------|---------|
| `.keyword` | Purple (#7d64ff) | `let`, `const`, `function`, `return`, `var`, `fn` |
| `.string` | Pink (#db2777) | `'text'`, `"text"` |
| `.function` | Blue (#2563eb) | Function names, method calls |
| `.comment` | Gray (#9ca3af) | `// comments`, `/* comments */` |
| `.property` | Dark gray (#4b5563) | Object properties, variables |
| `.number` | Green (#059669) | `123`, `0.5`, numeric literals |
| `.type` | Blue (#2563eb) | `u32`, `array`, type names |
| `.builtin` | Pink (#db2777) | `workgroup`, `@builtin`, special keywords |

## Usage in Your Files

### In Jekyll Markdown (.md files):
```markdown
{% include code-box-simple.html %}

<div class="code-box-simple">
  ...
</div>
```

### In HTML files:
Just paste the styles from `code-box-simple.html` into your `<head>` or `<style>` tag.

## Color Customization

Change these CSS variables:
```css
:root {
  --hl-keyword: #7d64ff;   /* Purple */
  --hl-string: #db2777;    /* Pink */
  --hl-function: #2563eb;  /* Blue */
  --hl-comment: #9ca3af;   /* Gray */
  --hl-number: #059669;    /* Green */
}
```

## Tips

1. **Manual highlighting**: Wrap each token in appropriate `<span class="...">` tags
2. **Keep it simple**: Don't over-highlight; plain text is fine for many elements
3. **Indentation**: Use 2 spaces for cleaner code blocks
4. **Line length**: Keep lines under 80 characters when possible for readability

## File Locations

- Component: `docs/_includes/code-box-simple.html`
- Examples: `docs/_includes/code-box-example.html`
- Guide: `docs/code-box-guide.md`
- Demo: `code-box-demo.html` (open in browser)
- Quick Ref: `docs/code-box-quick-reference.md` (this file)
