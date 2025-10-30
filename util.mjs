// not thrilled this uses idx, would rather do destructuring and not rely on idx/length
// would also like to see this done with a generator
export function combinations(obj) {
  const keys = Object.keys(obj);
  const values = Object.values(obj);

  function gen(values, idx) {
    if (idx === values.length) {
      return [{}]; // Base case: return an array with an empty object
    }

    const current = values[idx];
    const remaining = gen(values, idx + 1); // recurse

    const combos = [];
    for (const item of current) {
      for (const combination of remaining) {
        combos.push({ ...combination, [keys[idx]]: item }); // Combine the current item with each of the combinations from the rest
      }
    }
    return combos;
  }

  return gen(values, 0);
}

export const range = (min, max) => {
  const len = max - min + 1;
  const result = new Array(len);
  for (let i = 0; i < len; i++) {
    result[i] = min + i;
  }
  return result;
}

export const arrayProd = (arr) => {
  if (typeof arr === "number") {
    return arr;
  } else {
    return arr.reduce((a, b) => a * b);
  }
};

export function divRoundUp(x, y) {
  return Math.ceil(x / y);
}

export function bitreverse(i) {
  // https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
  let v = i >>> 0;
  let r = i >>> 0;
  let s = 31; // u32 -> 32 bits, minus 1

  for (v >>= 1; v; v >>= 1) {
    r <<= 1;
    r |= v & 1;
    s--;
  }
  r <<= s; // shift when v's highest bits are zero
  return r >>> 0;
}

export function fail(msg) {
  alert(msg);
}

export function datatypeToTypedArray(datatype) {
  switch (datatype) {
    case "f32":
      return Float32Array;
    case "i32":
      return Int32Array;
    case "u32":
    case "vec4u":
      return Uint32Array;
    case "u64":
      return BigUint64Array;
  }
  return undefined;
}

export function datatypeToBytes(datatype) {
  switch (datatype) {
    case "f32":
      return Float32Array.BYTES_PER_ELEMENT;
    case "i32":
      return Int32Array.BYTES_PER_ELEMENT;
    case "u32":
      return Uint32Array.BYTES_PER_ELEMENT;
    case "vec4u":
      return Uint32Array.BYTES_PER_ELEMENT * 4;
    case "u64":
      return BigUint64Array.BYTES_PER_ELEMENT;
  }
  return undefined;
}

// https://stackoverflow.com/questions/8896327/jquery-wait-delay-1-second-without-executing-code
export const delay = (millis) =>
  // eslint-disable-next-line no-unused-vars
  new Promise((resolve, reject) => {
    // eslint-disable-next-line no-unused-vars
    setTimeout((_) => resolve(), millis);
  });

// https://stackoverflow.com/questions/3665115/how-to-create-a-file-in-memory-for-user-to-download-but-not-through-server
export function download(content, mimeType, filename) {
  const a = document.createElement("a"); // Create "a" element
  if (mimeType == "application/json") {
    content = JSON.stringify(content);
  }
  const blob = new Blob([content], { type: mimeType }); // Create a blob (file-like object)
  const url = URL.createObjectURL(blob); // Create an object URL from blob
  a.setAttribute("href", url); // Set "a" element link
  a.setAttribute("download", filename); // Set download filename
  a.click(); // Start downloading
  URL.revokeObjectURL(url);
}

export function f32approxeq(reference, target) {
  const margin = 0;
  const scale = 0;
  const f32epsilon = 10000 * 1.192092896e-7;
  const functionallyZero = 0.01;
  function marginComparison(lhs, rhs, margin) {
    // Math.abs(lhs - rhs) <= margin, but allows infinity
    return lhs + margin >= rhs && rhs + margin >= lhs;
  }
  return (
    marginComparison(reference, target, margin) /* equal */ ||
    (marginComparison(0, reference, functionallyZero) &&
      marginComparison(
        0,
        target,
        functionallyZero
      )) /* close enough to zero */ ||
    marginComparison(
      /* within f32epsilon of each other */
      reference,
      target,
      f32epsilon * (scale + Math.abs(Number.isFinite(target) ? target : 0))
    )
  );
}

/**
 * Generates an array of integers evenly spaced on a logarithmic scale,
 * with each point rounded to the nearest multiple of 4.
 *
 * The function interpolates on a log10 scale and then converts the values
 * back before rounding. The returned array will include points corresponding
 * to the start and stop values, each subjected to the rounding rule.
 *
 * @param {number} start The starting integer of the sequence. Must be a positive number.
 * @param {number} stop The ending integer of the sequence. Must be a positive number.
 * @param {number} numPoints The total number of points to generate. Must be at least 2.
 * @returns {number[]} An array of integers representing the log-spaced sequence.
 * @throws {Error} If numPoints is less than 2, or if start/stop are not positive.
 */
export function logspaceRounded(start, stop, numPoints) {
  // --- 1. Validate Inputs ---
  if (numPoints < 2) {
    throw new Error("The number of points must be at least 2.");
  }
  if (start <= 0 || stop <= 0) {
    throw new Error(
      "Start and stop values must be positive for a logarithmic scale."
    );
  }

  /**
   * Helper function to round a number to the nearest multiple of 4.
   * @param {number} num The number to round.
   * @returns {number} The number rounded to the nearest multiple of 4.
   */
  const roundToNearestMultipleOf4 = (num) => {
    return Math.round(num / 4) * 4;
  };

  // --- 2. Perform Logarithmic Interpolation ---
  const result = [];
  const logStart = Math.log10(start);
  const logStop = Math.log10(stop);

  // The number of intervals is one less than the number of points.
  const step = (logStop - logStart) / (numPoints - 1);

  for (let i = 0; i < numPoints; i++) {
    // Calculate the intermediate point on the logarithmic scale.
    const logPoint = logStart + i * step;

    // Convert the point back to the original linear scale.
    const value = Math.pow(10, logPoint);

    // Round the value to the nearest multiple of 4 and add it to the array.
    result.push(roundToNearestMultipleOf4(value));
  }

  return result;
}

export function formatWGSL(wgslCode) {
  const lines = wgslCode.split("\n");
  const indent = "  ";
  let formattedLines = [];
  let indentLevel = 0;

  lines.forEach((line) => {
    /* Remove leading/trailing whitespace */
    const trimmedLine = line.trim();

    /** could combine multiple blank lines into one, but
     * that would mess up line numbering, so not doing that
     */

    const braceCount =
      (trimmedLine.match(/[{([]/g) || []).length -
      (trimmedLine.match(/[})\]]/g) || []).length;

    const pushLeft =
      /* lines like ") -> f32 {" */
      braceCount == 0 &&
      (trimmedLine.startsWith(")") ||
        trimmedLine.startsWith("]") ||
        trimmedLine.startsWith("}"))
        ? -1
        : 0;

    /* handle comments - pretty specific to my commenting style */
    const oneLineComment =
      trimmedLine.startsWith("//") ||
      (trimmedLine.startsWith("/*") && trimmedLine.endsWith("*/"));
    const midComment =
      /* I'm in the middle of a "documentation block" comment */
      trimmedLine.startsWith("* ") || trimmedLine == "*" || trimmedLine == "*/";
    const midCommentIndent = midComment ? " " : "";
    const inAComment = midComment || oneLineComment;

    if (braceCount > 0) {
      formattedLines.push(
        indent.repeat(indentLevel) + midCommentIndent + trimmedLine
      );
      if (inAComment == false) {
        /* only permanently change indent level if we're in code */
        indentLevel += braceCount;
      }
    } else if (braceCount < 0) {
      if (inAComment == false) {
        /* only permanently change indent level if we're in code */
        indentLevel += braceCount; /* adding a negative number */
      }
      if (indentLevel < 0) {
        indentLevel = 0;
      }
      formattedLines.push(
        indent.repeat(indentLevel) + midCommentIndent + trimmedLine
      );
    } else {
      formattedLines.push(
        indent.repeat(indentLevel + pushLeft) + midCommentIndent + trimmedLine
      );
    }
  });
  return formattedLines.join("\n");
}
