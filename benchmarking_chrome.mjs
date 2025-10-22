const isLocalhost =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:";

const modulePath =
  (isLocalhost ? window.location.origin : "https://gridwise-webgpu.github.io") +
  "/gridwise/benchmarking.mjs";

import(modulePath)
  .then(({ main }) => {
    main(navigator);
  })
  .catch((error) => {
    console.error("Error loading module", error);
  });

//import { main } from "http://localhost:8000/gridwise/benchmarking.mjs";
// main(navigator);
