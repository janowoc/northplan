// SPDX-FileCopyrightText: 2026 Jan Owoc
// SPDX-License-Identifier: AGPL-3.0-or-later

// The scenario form both pages share. It owns three globals: NorthplanJson, a JSON codec that
// keeps each number's text and each object's key order; northplanNumberText, the rule that
// turns what a number control holds into JSON number text; and northplanEditor, which builds
// the form's state and methods for a page's Alpine data and draws the form from the schema.
// It makes no request of its own: a page hands it an adapter, {call(name, argumentsText),
// file(name), runHoldsForm}, and every byte that crosses the network or the worker goes through
// that. When runHoldsForm is true, formHeld() is true while the page's running field is; the
// pages turn the form and its three open buttons off by it, and openForm and openFile do
// nothing then.

const NorthplanJson = (() => {
  const NUMBER = /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/y;
  const STRING = /"(?:[^"\\\u0000-\u001f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"/y;
  const SPACE = /[ \t\n\r]*/y;

  class JsonNumber {
    constructor(text) {
      this.text = text;
    }
  }

  function parse(text) {
    let at = 0;
    const fail = () => {
      throw new SyntaxError(`not one JSON value, at character ${at}`);
    };
    const skip = () => {
      SPACE.lastIndex = at;
      SPACE.exec(text);
      at = SPACE.lastIndex;
    };
    const token = (pattern) => {
      pattern.lastIndex = at;
      const found = pattern.exec(text);
      if (found === null) {
        fail();
      }
      at = pattern.lastIndex;
      return found[0];
    };
    const value = () => {
      skip();
      const c = text[at];
      if (c === '{') {
        at += 1;
        const map = new Map();
        skip();
        if (text[at] === '}') {
          at += 1;
          return map;
        }
        for (;;) {
          skip();
          const key = JSON.parse(token(STRING));
          skip();
          if (text[at] !== ':') {
            fail();
          }
          at += 1;
          if (map.has(key)) {
            fail();
          }
          map.set(key, value());
          skip();
          if (text[at] === ',') {
            at += 1;
          } else if (text[at] === '}') {
            at += 1;
            return map;
          } else {
            fail();
          }
        }
      }
      if (c === '[') {
        at += 1;
        const list = [];
        skip();
        if (text[at] === ']') {
          at += 1;
          return list;
        }
        for (;;) {
          list.push(value());
          skip();
          if (text[at] === ',') {
            at += 1;
          } else if (text[at] === ']') {
            at += 1;
            return list;
          } else {
            fail();
          }
        }
      }
      if (c === '"') {
        return JSON.parse(token(STRING));
      }
      if (text.startsWith('true', at)) {
        at += 4;
        return true;
      }
      if (text.startsWith('false', at)) {
        at += 5;
        return false;
      }
      if (text.startsWith('null', at)) {
        at += 4;
        return null;
      }
      return new JsonNumber(token(NUMBER));
    };
    const result = value();
    skip();
    if (at !== text.length) {
      fail();
    }
    return result;
  }

  function stringify(value) {
    if (value === null || value === true || value === false) {
      return String(value);
    }
    if (value instanceof JsonNumber) {
      return value.text;
    }
    if (typeof value === 'number') {
      if (!Number.isSafeInteger(value)) {
        throw new TypeError(`cannot write the number ${value} exactly`);
      }
      return String(value);
    }
    if (typeof value === 'string') {
      return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
      return '[' + value.map(stringify).join(',') + ']';
    }
    if (value instanceof Map) {
      const parts = [];
      for (const [key, item] of value) {
        parts.push(JSON.stringify(key) + ':' + stringify(item));
      }
      return '{' + parts.join(',') + '}';
    }
    throw new TypeError(`cannot write ${Object.prototype.toString.call(value)} as JSON`);
  }

  function toNumber(value) {
    return Number(value.text);
  }

  return { JsonNumber, parse, stringify, toNumber };
})();

function northplanNumberText(text) {
  const found = /^([+-]?)(\d*)(?:(\.)(\d*))?(?:([eE])([+-]?\d+))?$/.exec(text.trim());
  if (found === null) {
    return null;
  }
  const [, sign, whole, point, fraction, e, exponent] = found;
  if (whole === '' && (fraction === undefined || fraction === '')) {
    return null;
  }
  let out = sign === '-' ? '-' : '';
  out += whole === '' ? '0' : whole.replace(/^0+(?=\d)/, '');
  if (point !== undefined) {
    out += '.' + (fraction === '' ? '0' : fraction);
  }
  if (e !== undefined) {
    out += e + exponent;
  }
  if ((point !== undefined || e !== undefined) && !Number.isFinite(Number(out))) {
    return null;
  }
  return out;
}

const northplanEditor = (() => {
  const { JsonNumber } = NorthplanJson;
  const CHECK_DELAY_MS = 300;

  const style = document.createElement('style');
  style.textContent = `
    .editor-bar button { margin-right: 0.5rem; }
    .editor-state { opacity: 0.8; margin: 0.5rem 0; }
    .editor-problem, .editor-errors, .editor-notices {
      margin: 0.75rem 0; padding: 0.5rem 1rem; border-left: 3px solid #c0392b;
    }
    .editor-notices { border-left-color: #b9770e; }
    .editor-problem p, .editor-errors p { margin: 0.25rem 0; }
    .editor-problem ul, .editor-errors ul, .editor-notices { padding-left: 1.5rem; }
    #form { border: 1px solid rgba(128, 128, 128, 0.5); margin: 0.75rem 0; padding: 0.5rem 0.75rem; }
    #form .editor-field {
      margin: 0.35rem 0 0.35rem 0.75rem; padding-left: 0.6rem;
      border-left: 1px solid rgba(128, 128, 128, 0.35);
    }
    #form td.editor-field { margin: 0; padding: 0.1rem 0.2rem; border-left: 0; }
    #form .editor-key { font-weight: 600; margin-right: 0.5rem; }
    #form .editor-help { margin: 0.1rem 0 0.3rem; font-size: 0.8rem; opacity: 0.7; }
    #form .editor-field-errors {
      margin: 0.2rem 0 0; padding-left: 1.2rem; font-size: 0.85rem; color: #c0392b;
    }
    #form .editor-field-errors:empty { display: none; }
    #form .editor-value, #form .editor-name, #form .editor-new-name {
      font: inherit; padding: 0.15rem 0.3rem; max-width: 100%; box-sizing: border-box;
    }
    #form input[type=text].editor-value { width: 16rem; }
    #form .editor-matrix input.editor-value { width: 5rem; }
    #form button { margin: 0 0.25rem 0 0; padding: 0.1rem 0.6rem; font-size: 0.85rem; }
    #form .editor-choice { margin: 0.2rem 0; }
    #form .editor-choice span { margin: 0 0.75rem 0 0.2rem; }
    #form .editor-matrix { border-collapse: collapse; }
    #form .editor-matrix th { font-weight: 600; padding: 0.1rem 0.4rem; text-align: left; }
    #form .editor-raw { white-space: pre-wrap; word-break: break-all; }
    #form:disabled { opacity: 0.6; }
  `;
  document.head.appendChild(style);

  const read = (container, key) => (container instanceof Map ? container.get(key) : container[key]);
  const write = (container, key, value) => {
    if (container instanceof Map) {
      container.set(key, value);
    } else {
      container[key] = value;
    }
  };
  const dig = (value, ...path) => {
    for (const step of path) {
      if (value instanceof Map) {
        value = value.get(step);
      } else if (Array.isArray(value) && typeof step === 'number') {
        value = value[step];
      } else {
        return undefined;
      }
    }
    return value;
  };
  const clone = (value) => {
    if (value instanceof Map) {
      return new Map(Array.from(value, ([k, v]) => [k, clone(v)]));
    }
    return Array.isArray(value) ? value.map(clone) : value;
  };
  const shown = (value) => {
    if (value instanceof JsonNumber) {
      return value.text;
    }
    return value === null || value === undefined ? '' : String(value);
  };
  const plain = (loc) =>
    loc.map((step) => (step instanceof JsonNumber ? NorthplanJson.toNumber(step) : step));
  const dotted = (loc) => plain(loc).join('.');
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className !== undefined) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  };
  const button = (label, className, action) => {
    const node = make('button', className, label);
    node.type = 'button';
    node.addEventListener('click', action);
    return node;
  };
  const choiceSelect = (values, className) => {
    const select = make('select', className);
    for (const value of values) {
      const option = make('option', undefined, value);
      option.value = value;
      select.appendChild(option);
    }
    return select;
  };
  const argumentsOf = (entries) => NorthplanJson.stringify(new Map(entries));

  return function northplanEditor(adapter) {
    let schema = null;
    let mapping = null;
    let version = 0;
    let timer = null;
    let gridPaths = [];
    let errorList = [];
    let registry = new Map();
    let gridSelect = null;
    let gridAdd = null;
    let ui = null;

    // --- the schema ---------------------------------------------------------------

    function resolve(node) {
      for (let hops = 0; node instanceof Map && typeof node.get('$ref') === 'string'; hops += 1) {
        const name = node.get('$ref').replace('#/$defs/', '');
        node = dig(schema, '$defs', name);
        if (hops > 50) {
          break;
        }
      }
      return node;
    }

    function classify(node) {
      if (!(node instanceof Map)) {
        return { kind: 'raw', res: new Map(), nullable: false, get: () => undefined };
      }
      const options = node.get('anyOf');
      if (Array.isArray(options) && options.length === 2) {
        const isNull = (option) => option instanceof Map && option.get('type') === 'null';
        const at = options.findIndex(isNull);
        if (at !== -1) {
          return { ...classify(options[1 - at]), nullable: true };
        }
      }
      const res = resolve(node);
      if (!(res instanceof Map)) {
        return { kind: 'raw', res: new Map(), nullable: false, get: () => undefined };
      }
      const get = (key) => (res.has(key) ? res.get(key) : node.get(key));
      const info = { kind: 'raw', res, nullable: false, get };
      const type = get('type');
      const extra = resolve(get('additionalProperties'));
      const widget = get('x-northplan-widget');
      const numeric = (n) => n instanceof Map && (n.get('type') === 'number' || n.get('type') === 'integer');
      if (widget === 'correlation') {
        info.kind = 'correlation';
      } else if (widget === 'grid') {
        info.kind = 'grid';
      } else if (get('x-northplan-defines') === 'asset-class') {
        info.kind = 'assetClasses';
        info.itemRes = extra;
      } else if (res.has('properties')) {
        info.kind = 'model';
      } else if (
        type === 'object' &&
        get('propertyNames') instanceof Map &&
        extra instanceof Map &&
        extra.get('additionalProperties') instanceof Map
      ) {
        info.kind = 'mapOfMaps';
        info.names = get('propertyNames').get('enum') || [];
        info.inner = classify(get('additionalProperties'));
      } else if (type === 'object' && numeric(extra)) {
        info.kind = 'numberMap';
      } else if (type === 'array' && resolve(get('items'))?.get('properties') instanceof Map) {
        info.kind = 'list';
        info.itemRes = resolve(get('items'));
      } else if (
        type === 'array' &&
        get('uniqueItems') === true &&
        Array.isArray(resolve(get('items'))?.get('enum'))
      ) {
        info.kind = 'pick';
        info.choices = resolve(get('items')).get('enum');
      } else if (type === 'string' && Array.isArray(get('enum'))) {
        info.kind = 'select';
        info.choices = get('enum');
      } else if (type === 'string' && get('x-northplan-ref') === 'person') {
        info.kind = 'personSelect';
      } else if (type === 'string' && get('x-northplan-defines') === 'person') {
        info.kind = 'personId';
      } else if (type === 'string') {
        info.kind = 'text';
      } else if (type === 'number' || type === 'integer') {
        info.kind = 'number';
      } else if (type === 'boolean') {
        info.kind = 'checkbox';
      }
      return info;
    }

    function skeleton(res) {
      const found = new Map();
      const required = res.get('required') || [];
      for (const [key, node] of res.get('properties')) {
        if (!required.includes(key)) {
          continue;
        }
        const info = classify(node);
        if (info.kind === 'model') {
          found.set(key, skeleton(info.res));
        } else if (['numberMap', 'mapOfMaps', 'assetClasses', 'grid'].includes(info.kind)) {
          found.set(key, new Map());
        } else if (['list', 'pick', 'correlation'].includes(info.kind)) {
          found.set(key, []);
        }
      }
      return found;
    }

    function initialValue(node, info) {
      if (node.has('default') && node.get('default') !== null) {
        return clone(node.get('default'));
      }
      if (info.kind === 'model') {
        return skeleton(info.res);
      }
      const type = info.get('type');
      if (type === 'object') {
        return new Map();
      }
      return type === 'array' ? [] : '';
    }

    // --- the mapping --------------------------------------------------------------

    function personIds() {
      const found = [];
      const persons = dig(mapping, 'household', 'persons');
      for (const person of Array.isArray(persons) ? persons : []) {
        const id = dig(person, 'id');
        if (typeof id === 'string' && !found.includes(id)) {
          found.push(id);
        }
      }
      return found;
    }

    function holders(id) {
      const persons = dig(mapping, 'household', 'persons');
      return (Array.isArray(persons) ? persons : []).filter((p) => dig(p, 'id') === id).length;
    }

    function assetClassNames() {
      const classes = dig(mapping, 'assumptions', 'asset_classes');
      return classes instanceof Map ? Array.from(classes.keys()) : [];
    }

    // --- the form's state ---------------------------------------------------------

    function failure(name, reply) {
      const body = reply.body;
      let detail;
      if (body instanceof Map && typeof body.get('detail') === 'string') {
        detail = body.get('detail');
      } else if (typeof body === 'string') {
        detail = body;
      } else {
        detail = NorthplanJson.stringify(body);
      }
      return `The ${name} call failed (${reply.status}): ${detail}`;
    }

    const drawFailure = (error) => `The form could not be drawn: ${String(error)}`;

    function errorsOf(list) {
      return (Array.isArray(list) ? list : []).map((entry) => ({
        loc: plain(entry.get('loc')),
        message: entry.get('message'),
      }));
    }

    const lineOf = (e) => (e.loc.length ? `${e.loc.join('.')}: ${e.message}` : e.message);

    function placeErrors() {
      for (const element of registry.values()) {
        element.lastElementChild.replaceChildren();
      }
      for (const e of errorList) {
        for (let n = e.loc.length; n >= 1; n -= 1) {
          const element = registry.get(JSON.stringify(e.loc.slice(0, n)));
          if (element !== undefined) {
            element.lastElementChild.appendChild(make('li', undefined, e.message));
            break;
          }
        }
      }
    }

    function refreshGrid() {
      if (gridSelect === null) {
        return;
      }
      const grid = dig(mapping, 'grid');
      const options = gridPaths.filter((path) => !(grid instanceof Map && grid.has(path)));
      const kept = gridSelect.value;
      gridSelect.replaceChildren();
      for (const path of options) {
        const option = make('option', undefined, path);
        option.value = path;
        gridSelect.appendChild(option);
      }
      if (options.includes(kept)) {
        gridSelect.value = kept;
      }
      gridAdd.disabled = options.length === 0;
    }

    function changed() {
      version += 1;
      ui.formYaml = null;
      ui.formState = 'checking';
      ui.formProblem = null;
      ui.formReferences = [];
      clearTimeout(timer);
      timer = setTimeout(check, CHECK_DELAY_MS);
    }

    async function check() {
      timer = null;
      const v = version;
      const host = ui;
      let text;
      try {
        text = argumentsOf([['mapping', mapping]]);
      } catch (error) {
        host.formProblem = `The check call failed (client): ${error.message}`;
        host.formYaml = null;
        host.formState = 'errors';
        return;
      }
      const pending = adapter
        .call('check', text)
        .catch((error) => ({ status: 'error', body: String(error) }));
      adapter
        .call('grid-paths', text)
        .then((reply) => {
          if (version === v && reply.status === 200 && Array.isArray(reply.body)) {
            gridPaths = reply.body.filter((path) => typeof path === 'string');
            refreshGrid();
          }
        })
        .catch(() => {});
      const reply = await pending;
      if (version !== v) {
        return;
      }
      if (reply.status === 200 && reply.body instanceof Map) {
        errorList = errorsOf(reply.body.get('errors'));
        host.formErrors = errorList.map(lineOf);
        placeErrors();
        host.formYaml = errorList.length === 0 ? reply.body.get('yaml') : null;
        host.formState = errorList.length === 0 ? 'clean' : 'errors';
      } else {
        errorList = [];
        host.formErrors = [];
        placeErrors();
        host.formProblem = failure('check', reply);
        host.formYaml = null;
        host.formState = 'errors';
      }
    }

    async function pythonEdit(name, entries) {
      const host = ui;
      host.formBusy = true;
      try {
        const v = version;
        const reply = await adapter.call(name, argumentsOf([['mapping', mapping], ...entries]));
        if (version !== v) {
          return;
        }
        if (reply.status === 200) {
          mapping = reply.body;
          if (redraw(host)) {
            changed();
          }
        } else if (reply.status === 409 && reply.body instanceof Map) {
          host.formProblem = reply.body.get('detail');
          host.formReferences = (reply.body.get('references') || []).map(dotted);
          redraw(host);
        } else {
          host.formProblem = failure(name, reply);
          redraw(host);
        }
      } finally {
        host.formBusy = false;
      }
    }

    // --- drawing ------------------------------------------------------------------

    function wrapper(path, tag) {
      const element = make(tag || 'div', 'editor-field');
      const loc = JSON.stringify(path);
      element.setAttribute('data-loc', loc);
      element.appendChild(make('ul', 'editor-field-errors'));
      registry.set(loc, element);
      return element;
    }

    function add(parent, child) {
      if (parent.classList.contains('editor-field')) {
        parent.insertBefore(child, parent.lastElementChild);
      } else {
        parent.appendChild(child);
      }
      return child;
    }

    function redraw(host) {
      try {
        render();
        return true;
      } catch (error) {
        version += 1;
        clearTimeout(timer);
        timer = null;
        host.formYaml = null;
        host.formState = 'errors';
        host.formProblem = drawFailure(error);
        return false;
      }
    }

    function restructure() {
      if (redraw(ui)) {
        changed();
      }
    }

    function rawCode(parent, value) {
      let text = '';
      if (value !== undefined) {
        try {
          text = NorthplanJson.stringify(value);
        } catch (error) {
          text = String(value);
        }
      }
      add(parent, make('code', 'editor-raw', text));
    }

    function numberInput(value, onValue) {
      const input = make('input', 'editor-value');
      input.type = 'text';
      input.setAttribute('inputmode', 'decimal');
      input.value = shown(value);
      input.addEventListener('input', () => {
        const text = northplanNumberText(input.value);
        onValue(text !== null ? new JsonNumber(text) : input.value, input.value);
        changed();
      });
      return input;
    }

    function drawScalar(parent, info, container, key, disabled) {
      const value = read(container, key);
      if (value instanceof Map || Array.isArray(value)) {
        rawCode(parent, value);
        return;
      }
      let input;
      if (info.kind === 'number') {
        input = numberInput(value, (v) => write(container, key, v));
      } else if (info.kind === 'text') {
        input = make('input', 'editor-value');
        input.type = 'text';
        input.value = shown(value);
        input.addEventListener('input', () => {
          write(container, key, input.value);
          changed();
        });
      } else if (info.kind === 'select' || info.kind === 'personSelect') {
        const choices = info.kind === 'select' ? [...info.choices] : personIds();
        if (typeof value === 'string' && value !== '' && !choices.includes(value)) {
          choices.push(value);
        }
        input = choiceSelect(['', ...choices], 'editor-value');
        input.value = typeof value === 'string' ? value : '';
        input.addEventListener('change', () => {
          write(container, key, input.value);
          changed();
        });
      } else if (info.kind === 'checkbox') {
        input = make('input', 'editor-value');
        input.type = 'checkbox';
        input.checked = value === true;
        input.indeterminate = value === undefined;
        input.addEventListener('change', () => {
          write(container, key, input.checked);
          changed();
        });
      } else {
        input = make('input', 'editor-value');
        input.type = 'text';
        input.value = shown(value);
        input.addEventListener('change', () => {
          const old = read(container, key);
          if (input.value === old) {
            return;
          }
          if (typeof old === 'string' && holders(old) === 1) {
            pythonEdit('rename', [
              ['kind', 'person'],
              ['old', old],
              ['new', input.value],
            ]);
          } else {
            write(container, key, input.value);
            restructure();
          }
        });
      }
      input.disabled = disabled;
      add(parent, input);
    }

    function drawBody(parent, res, obj, path) {
      const properties = res.get('properties');
      const required = res.get('required') || [];
      const exactlyOne = res.get('x-northplan-exactly-one');
      let chosen = null;
      if (Array.isArray(exactlyOne)) {
        chosen =
          exactlyOne.find((k) => obj.has(k) && obj.get(k) !== null) ??
          exactlyOne.find((k) => obj.has(k)) ??
          null;
        const choice = make('div', 'editor-choice');
        const group = JSON.stringify(path);
        for (const k of exactlyOne) {
          const radio = make('input');
          radio.type = 'radio';
          radio.name = group;
          radio.value = k;
          radio.checked = k === chosen;
          radio.addEventListener('change', () => {
            for (const other of exactlyOne) {
              obj.delete(other);
            }
            obj.set(k, '');
            restructure();
          });
          choice.appendChild(radio);
          choice.appendChild(make('span', undefined, k));
        }
        add(parent, choice);
      }
      for (const [key, node] of properties) {
        if (Array.isArray(exactlyOne)) {
          if (key === chosen) {
            drawProperty(parent, obj, key, node, [...path, key], false, true);
          }
        } else {
          drawProperty(parent, obj, key, node, [...path, key], required.includes(key), false);
        }
      }
    }

    function drawProperty(parent, obj, key, node, path, required, forced) {
      const info = classify(node);
      const element = add(parent, wrapper(path));
      const optional = !required && !forced;
      const present = obj.has(key);
      if (optional) {
        const box = make('input', 'editor-present');
        box.type = 'checkbox';
        box.checked = present;
        box.addEventListener('change', () => {
          if (box.checked) {
            obj.set(key, initialValue(node, info));
          } else {
            obj.delete(key);
          }
          restructure();
        });
        add(element, box);
      }
      add(element, make('code', 'editor-key', key));
      const nullBox = required && info.nullable;
      if (nullBox) {
        const box = make('input', 'editor-null');
        box.type = 'checkbox';
        box.checked = obj.get(key) === null;
        box.addEventListener('change', () => {
          obj.set(key, box.checked ? null : '');
          restructure();
        });
        add(element, box);
        add(element, make('span', undefined, 'none'));
      }
      if (typeof node.get('description') === 'string') {
        add(element, make('p', 'editor-help', node.get('description')));
      }
      if (optional && !present) {
        return;
      }
      drawContent(element, info, obj, key, path, key, nullBox && obj.get(key) === null);
    }

    function drawContent(parent, info, container, key, path, label, disabled) {
      const value = read(container, key);
      switch (info.kind) {
        case 'text':
        case 'number':
        case 'select':
        case 'personSelect':
        case 'checkbox':
        case 'personId':
          drawScalar(parent, info, container, key, disabled);
          break;
        case 'model':
          if (value instanceof Map) {
            drawBody(parent, info.res, value, path);
          } else {
            rawCode(parent, value);
          }
          break;
        case 'list':
          if (Array.isArray(value)) {
            drawList(parent, info, value, path, label);
          } else {
            rawCode(parent, value);
          }
          break;
        case 'numberMap':
          if (value instanceof Map) {
            drawNumberRows(parent, info, value, path);
          } else {
            rawCode(parent, value);
          }
          break;
        case 'mapOfMaps':
          if (value instanceof Map) {
            drawAccounts(parent, info, value, path);
          } else {
            rawCode(parent, value);
          }
          break;
        case 'assetClasses':
          if (value instanceof Map) {
            drawClasses(parent, info, value, path);
          } else {
            rawCode(parent, value);
          }
          break;
        case 'correlation':
          if (Array.isArray(value) && value.every((row) => Array.isArray(row))) {
            drawMatrix(parent, value, path);
          } else {
            rawCode(parent, value);
          }
          break;
        case 'grid':
          if (value instanceof Map) {
            drawGrid(parent, value, path);
          } else {
            rawCode(parent, value);
          }
          break;
        case 'pick':
          if (Array.isArray(value)) {
            drawPick(parent, info, value, path);
          } else {
            rawCode(parent, value);
          }
          break;
        default:
          rawCode(parent, value);
      }
    }

    function drawList(parent, info, items, path, label) {
      const personList = path.length === 2 && path[0] === 'household' && path[1] === 'persons';
      items.forEach((item, i) => {
        const element = add(parent, wrapper([...path, i]));
        add(element, make('code', 'editor-key', `${label}[${i}]`));
        add(
          element,
          button('Remove', 'editor-remove', () => {
            const id = dig(item, 'id');
            if (personList && typeof id === 'string' && holders(id) === 1) {
              pythonEdit('remove', [
                ['kind', 'person'],
                ['name', id],
              ]);
            } else {
              items.splice(i, 1);
              restructure();
            }
          }),
        );
        if (item instanceof Map) {
          drawBody(element, info.itemRes, item, [...path, i]);
        } else {
          rawCode(element, item);
        }
      });
      add(
        parent,
        button('Add', 'editor-add', () => {
          items.push(skeleton(info.itemRes));
          restructure();
        }),
      );
    }

    function possibleKeys(info) {
      const kind = info.get('x-northplan-keys');
      if (kind === 'person') {
        return personIds();
      }
      if (kind === 'asset-class') {
        return assetClassNames();
      }
      const names = info.get('propertyNames');
      return names instanceof Map && Array.isArray(names.get('enum')) ? names.get('enum') : [];
    }

    function drawNumberRows(parent, info, map, path) {
      const keys = [...map.keys(), ...possibleKeys(info).filter((k) => !map.has(k))];
      for (const key of keys) {
        const row = add(parent, wrapper([...path, key]));
        add(row, make('code', 'editor-key', key));
        add(
          row,
          numberInput(map.get(key), (value, text) => {
            if (text.trim() === '') {
              map.delete(key);
            } else {
              map.set(key, value);
            }
          }),
        );
      }
    }

    function drawAccounts(parent, info, accounts, path) {
      for (const [account, weights] of accounts) {
        const element = add(parent, wrapper([...path, account]));
        add(element, make('code', 'editor-key', account));
        add(
          element,
          button('Remove', 'editor-remove', () => {
            accounts.delete(account);
            restructure();
          }),
        );
        if (weights instanceof Map) {
          drawNumberRows(element, info.inner, weights, [...path, account]);
        } else {
          rawCode(element, weights);
        }
      }
      const choices = info.names.filter((name) => !accounts.has(name));
      const select = add(parent, choiceSelect(choices, 'editor-add-choice'));
      const adder = add(
        parent,
        button('Add', 'editor-add', () => {
          accounts.set(select.value, new Map());
          restructure();
        }),
      );
      adder.disabled = choices.length === 0;
    }

    function drawClasses(parent, info, classes, path) {
      const names = Array.from(classes.keys());
      names.forEach((name, i) => {
        const element = add(parent, wrapper([...path, name]));
        const input = make('input', 'editor-name');
        input.type = 'text';
        input.value = name;
        input.addEventListener('change', () => {
          if (input.value !== name) {
            pythonEdit('rename', [
              ['kind', 'asset-class'],
              ['old', name],
              ['new', input.value],
            ]);
          }
        });
        add(element, input);
        const up = add(
          element,
          button('Up', 'editor-up', () =>
            pythonEdit('move-asset-class', [
              ['name', name],
              ['index', i - 1],
            ]),
          ),
        );
        up.disabled = i === 0;
        const down = add(
          element,
          button('Down', 'editor-down', () =>
            pythonEdit('move-asset-class', [
              ['name', name],
              ['index', i + 1],
            ]),
          ),
        );
        down.disabled = i === names.length - 1;
        add(
          element,
          button('Remove', 'editor-remove', () =>
            pythonEdit('remove', [
              ['kind', 'asset-class'],
              ['name', name],
            ]),
          ),
        );
        const cls = classes.get(name);
        if (cls instanceof Map && info.itemRes instanceof Map) {
          drawBody(element, info.itemRes, cls, [...path, name]);
        } else {
          rawCode(element, cls);
        }
      });
      const fresh = add(parent, make('input', 'editor-new-name'));
      fresh.type = 'text';
      add(
        parent,
        button('Add asset class', 'editor-add', () =>
          pythonEdit('add-asset-class', [['name', fresh.value]]),
        ),
      );
    }

    function drawMatrix(parent, matrix, path) {
      const names = assetClassNames();
      const table = add(parent, make('table', 'editor-matrix'));
      const head = table.appendChild(make('tr'));
      head.appendChild(make('th'));
      names.forEach((name) => head.appendChild(make('th', undefined, name)));
      const inputs = matrix.map(() => []);
      matrix.forEach((row, i) => {
        const tr = table.appendChild(make('tr'));
        tr.appendChild(make('th', undefined, names[i] ?? ''));
        row.forEach((cell, j) => {
          const td = wrapper([...path, i, j], 'td');
          tr.appendChild(td);
          const input = numberInput(cell, (value) => {
            matrix[i][j] = value;
            if (Array.isArray(matrix[j]) && j < matrix.length && i < matrix[j].length) {
              matrix[j][i] = value;
              if (inputs[j][i] !== undefined) {
                inputs[j][i].value = shown(value);
              }
            }
          });
          input.readOnly = i === j;
          inputs[i][j] = input;
          add(td, input);
        });
      });
    }

    function drawGrid(parent, grid, path) {
      for (const [key, values] of grid) {
        const element = add(parent, wrapper([...path, key]));
        add(element, make('code', 'editor-key', key));
        add(
          element,
          button('Remove', 'editor-remove', () => {
            grid.delete(key);
            restructure();
          }),
        );
        if (!Array.isArray(values)) {
          rawCode(element, values);
          continue;
        }
        const input = make('input', 'editor-value');
        input.type = 'text';
        input.value = values.map(shown).join(', ');
        input.addEventListener('input', () => {
          if (input.value.trim() === '') {
            grid.set(key, []);
          } else {
            grid.set(
              key,
              input.value.split(',').map((piece) => {
                const text = northplanNumberText(piece);
                return text !== null ? new JsonNumber(text) : piece.trim();
              }),
            );
          }
          changed();
        });
        add(element, input);
      }
      gridSelect = add(parent, make('select', 'editor-add-choice'));
      gridAdd = add(
        parent,
        button('Add', 'editor-add', () => {
          grid.set(gridSelect.value, []);
          restructure();
        }),
      );
      refreshGrid();
    }

    function drawPick(parent, info, items, path) {
      items.forEach((item, i) => {
        const element = add(parent, wrapper([...path, i]));
        const swap = (other) => {
          [items[i], items[other]] = [items[other], items[i]];
          restructure();
        };
        add(element, button('Up', 'editor-up', () => swap(i - 1))).disabled = i === 0;
        add(element, button('Down', 'editor-down', () => swap(i + 1))).disabled =
          i === items.length - 1;
        add(
          element,
          button('Remove', 'editor-remove', () => {
            items.splice(i, 1);
            restructure();
          }),
        );
        add(element, make('span', 'editor-value', shown(item)));
      });
      const choices = info.choices.filter((choice) => !items.includes(choice));
      const select = add(parent, choiceSelect(choices, 'editor-add-choice'));
      const adder = add(
        parent,
        button('Add', 'editor-add', () => {
          items.push(select.value);
          restructure();
        }),
      );
      adder.disabled = choices.length === 0;
    }

    function render() {
      const form = ui.$refs.form;
      form.replaceChildren();
      registry = new Map();
      gridSelect = null;
      gridAdd = null;
      if (mapping instanceof Map) {
        drawBody(form, schema, mapping, []);
      }
      placeErrors();
    }

    // --- opening and closing ------------------------------------------------------

    function close(host) {
      version += 1;
      clearTimeout(timer);
      timer = null;
      mapping = null;
      errorList = [];
      registry = new Map();
      gridSelect = null;
      gridAdd = null;
      host.$refs.form.replaceChildren();
      host.formOpen = false;
      host.formYaml = null;
      host.formErrors = [];
      host.formNotices = [];
      host.formReferences = [];
      host.formProblem = null;
      host.formState = 'clean';
    }

    async function openText(host, text) {
      ui = host;
      host.formBusy = true;
      host.formErrors = [];
      host.formNotices = [];
      host.formProblem = null;
      host.formReferences = [];
      try {
        if (schema === null) {
          const reply = await adapter.call('schema', '{}');
          if (reply.status !== 200 || !(reply.body instanceof Map)) {
            host.formProblem = failure('schema', reply);
            return;
          }
          schema = reply.body;
        }
        const reply = await adapter.call('parse', argumentsOf([['text', text]]));
        if (reply.status !== 200 || !(reply.body instanceof Map)) {
          host.formProblem = failure('parse', reply);
          return;
        }
        const result = reply.body;
        if (!(result.get('mapping') instanceof Map)) {
          host.formErrors = errorsOf(result.get('errors')).map(lineOf);
          return;
        }
        mapping = result.get('mapping');
        errorList = [];
        gridPaths = [];
        host.formNotices = result.get('notices');
        host.formOpen = true;
        version += 1;
        host.formState = 'checking';
        try {
          render();
        } catch (error) {
          close(host);
          host.formProblem = drawFailure(error);
          return;
        }
        check();
      } catch (error) {
        host.formProblem = String(error);
      } finally {
        host.formBusy = false;
      }
    }

    return {
      formOpen: false,
      formBusy: false,
      formState: 'clean',
      formErrors: [],
      formNotices: [],
      formReferences: [],
      formProblem: null,
      formYaml: null,
      openForm() {
        const host = this;
        if (host.formHeld()) {
          return undefined;
        }
        return openText(host, host.scenario);
      },
      async openFile(name) {
        const host = this;
        ui = host;
        if (host.formBusy || host.formHeld()) {
          return;
        }
        host.formBusy = true;
        host.formErrors = [];
        host.formNotices = [];
        host.formProblem = null;
        host.formReferences = [];
        let text;
        try {
          text = await adapter.file(name);
        } catch (error) {
          host.formProblem = `Could not open the file: ${error.message}`;
          host.formBusy = false;
          return;
        }
        await openText(host, text);
      },
      useYaml() {
        const host = this;
        if (!host.formOpen || host.formBusy || host.formYaml === null) {
          return;
        }
        host.scenario = host.formYaml;
        close(host);
      },
      downloadYaml() {
        const host = this;
        if (!host.formOpen || host.formBusy || host.formYaml === null) {
          return;
        }
        const name = dig(mapping, 'name');
        const url = URL.createObjectURL(new Blob([host.formYaml], { type: 'application/yaml' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = typeof name === 'string' && name !== '' ? `${name}.yaml` : 'scenario.yaml';
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      },
      discardForm() {
        const host = this;
        if (!host.formBusy) {
          close(host);
        }
      },
      formHeld() {
        return adapter.runHoldsForm === true && this.running === true;
      },
      formStateText() {
        if (this.formHeld()) {
          return 'The form is off while the scenario runs.';
        }
        if (this.formBusy) {
          return 'Working…';
        }
        if (this.formState === 'checking') {
          return 'Checking…';
        }
        if (this.formState === 'clean') {
          return 'No errors.';
        }
        return 'The scenario has errors; Run, Back to YAML and Download YAML stay off until they are fixed.';
      },
    };
  };
})();
