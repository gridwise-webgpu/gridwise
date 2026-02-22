# Simple Code Box Component (k6.io Style)

A clean, sidebar-free code box component inspired by k6.io's design aesthetic.

## Features

- ✨ Clean white borders with subtle shadows
- 🎨 Beautiful syntax highlighting (purple keywords, pink strings, blue functions)
- 📱 Responsive and mobile-friendly
- 🎯 No sidebar - just clean code display
- 🔧 Easy to customize colors

## Installation

1. The component is located in `docs/_includes/code-box-simple.html`
2. Include it in your markdown or HTML files using Jekyll's include syntax

## Usage

### Basic Usage

```html
<!-- Include the component first (do this once per page) -->
{% include code-box-simple.html %}

<!-- Then use the code box anywhere -->
<div class="code-box-simple">
  <div class="code-box-header">Your Title Here</div>
  <div class="code-box-content">
    <pre><code>Your code here</code></pre>
  </div>
</div>
```

### With Syntax Highlighting

Use these span classes for syntax highlighting:

- `.keyword` - for keywords (purple: `let`, `const`, `function`, `return`, etc.)
- `.string` - for strings (pink: `'hello'`, `"world"`)
- `.function` - for function names (blue: `connect()`, `log()`)
- `.comment` - for comments (gray, italic: `// comment`)
- `.property` - for properties (dark gray: object properties)
- `.number` - for numbers (green: `123`, `0.5`)
- `.type` - for types (blue: `u32`, `array`)
- `.builtin` - for builtins (pink: `workgroup`, `@builtin`)

### Example: JavaScript

```html
<div class="code-box-simple">
  <div class="code-box-header">WebSocket Connection</div>
  <div class="code-box-content">
    <pre><code><span class="keyword">export default function</span>() {
  <span class="keyword">let</span> url = <span class="string">'ws://echo.websocket.org'</span>
  <span class="keyword">let</span> params = { <span class="property">tags</span>: { <span class="property">my_tag</span>: <span class="string">'hello'</span> } }

  <span class="keyword">let</span> res = ws.<span class="function">connect</span>(url, params, <span class="keyword">function</span>(socket) {
    socket.<span class="function">on</span>(<span class="string">'open'</span>, <span class="keyword">function</span>() {
      console.<span class="function">log</span>(<span class="string">'connected'</span>)
    })
  })
}</code></pre>
  </div>
</div>
```

### Example: WGSL

```html
<div class="code-box-simple">
  <div class="code-box-header">WGSL Shader Code</div>
  <div class="code-box-content">
    <pre><code><span class="keyword">var</span>&lt;<span class="builtin">workgroup</span>&gt; <span class="property">wg_data</span>: <span class="type">array</span>&lt;<span class="type">u32</span>, <span class="number">256</span>&gt;;

<span class="keyword">fn</span> <span class="function">myFunction</span>(<span class="property">x</span>: <span class="type">u32</span>) -&gt; <span class="type">u32</span> {
  <span class="comment">// Perform calculation</span>
  <span class="keyword">return</span> <span class="property">x</span> * <span class="number">2</span>;
}</code></pre>
  </div>
</div>
```

## Customization

### Changing Colors

Edit the CSS variables in `code-box-simple.html`:

```css
:root {
  --code-border: #e5e7eb;        /* Border color */
  --code-bg: #ffffff;            /* Background */
  --code-title-bg: #f9fafb;      /* Header background */
  
  /* Syntax highlighting colors */
  --hl-keyword: #7d64ff;         /* Purple for keywords */
  --hl-string: #db2777;          /* Pink for strings */
  --hl-function: #2563eb;        /* Blue for functions */
  --hl-comment: #9ca3af;         /* Gray for comments */
  --hl-property: #4b5563;        /* Dark gray for properties */
  --hl-number: #059669;          /* Green for numbers */
}
```

### Removing the Header

If you don't want a header, just omit the `.code-box-header` div:

```html
<div class="code-box-simple">
  <div class="code-box-content">
    <pre><code>Your code here</code></pre>
  </div>
</div>
```

### Adding Multiple Code Boxes

You only need to include the component once per page, then use multiple code boxes:

```html
{% include code-box-simple.html %}

<div class="code-box-simple">
  <div class="code-box-header">Example 1</div>
  <div class="code-box-content">
    <pre><code>Code here</code></pre>
  </div>
</div>

<div class="code-box-simple">
  <div class="code-box-header">Example 2</div>
  <div class="code-box-content">
    <pre><code>More code here</code></pre>
  </div>
</div>
```

## Tips

1. **Manual Highlighting**: Syntax highlighting requires manually wrapping code in `<span>` tags. For large code blocks, consider using a syntax highlighter library.

2. **Line Width**: Long lines will scroll horizontally (overflow-x: auto). Keep code lines reasonable or add line breaks.

3. **Consistent Indentation**: Use 2 spaces for indentation to match the style.

4. **Testing**: Preview your page to ensure the highlighting looks correct.

## Browser Support

Works in all modern browsers that support CSS custom properties (CSS variables):
- Chrome/Edge 49+
- Firefox 31+
- Safari 9.1+

## Questions?

See the examples in `docs/_includes/code-box-example.html` for more usage patterns.
