const { WASI } = require("node:wasi");
const { readFile } = require("node:fs/promises");

async function main() {
  const [binary, ...args] = process.argv.slice(2);
  const wasi = new WASI({
    version: "preview1",
    args: [binary, ...args],
    env: process.env,
    preopens: { "/": process.cwd() },
  });
  const module = await WebAssembly.compile(await readFile(binary));
  const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
  wasi.start(instance);
}

main().catch((error) => {
  process.stderr.write(`${error.stack}\n`);
  process.exitCode = 1;
});
