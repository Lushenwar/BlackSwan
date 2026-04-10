/**
 * Manual Jest mock for the `vscode` module.
 *
 * Covers every VS Code API symbol used by diagnostics.ts, codelens.ts, and
 * orchestrator.ts. Value classes (Range, Position, Uri, etc.) are real
 * implementations so tests can assert on field values. API namespaces
 * (window, languages, commands, workspace) use jest.fn() stubs that tests
 * can override per-case with mockImplementation / mockResolvedValue.
 *
 * Key design:
 *   withProgress — actually invokes the task callback (with a fake progress
 *   object and cancellation token) so orchestrator logic runs in unit tests.
 *   Tests that need to pause mid-run can override this with mockImplementation.
 */

// ---------------------------------------------------------------------------
// Enums / constants
// ---------------------------------------------------------------------------

export const DiagnosticSeverity = {
  Error:       0,
  Warning:     1,
  Information: 2,
  Hint:        3,
} as const;

export const CodeActionKind = {
  Empty:    { value: "" },
  QuickFix: { value: "quickfix" },
  Refactor: { value: "refactor" },
  Source:   { value: "source" },
} as const;

export const ProgressLocation = {
  SourceControl:  1,
  Window:         10,
  Notification:   15,
} as const;

export const ViewColumn = {
  Active:  -1,
  Beside:  -2,
  One:      1,
  Two:      2,
  Three:    3,
} as const;

// ---------------------------------------------------------------------------
// Value classes — real lightweight implementations for field assertions.
// ---------------------------------------------------------------------------

export class Position {
  constructor(
    public readonly line: number,
    public readonly character: number,
  ) {}
}

export class Range {
  public readonly start: Position;
  public readonly end: Position;

  constructor(
    startLineOrPosition: number | Position,
    startCharacterOrEnd?: number | Position,
    endLine?: number,
    endCharacter?: number,
  ) {
    if (typeof startLineOrPosition === "number") {
      this.start = new Position(startLineOrPosition, startCharacterOrEnd as number);
      this.end   = new Position(endLine as number, endCharacter as number);
    } else {
      this.start = startLineOrPosition;
      this.end   = startCharacterOrEnd as Position;
    }
  }
}

export class Location {
  constructor(
    public readonly uri: Uri,
    public readonly range: Range,
  ) {}
}

export class DiagnosticRelatedInformation {
  constructor(
    public readonly location: Location,
    public readonly message: string,
  ) {}
}

export class Diagnostic {
  public source?: string;
  public code?: string | number | { value: string | number; target: Uri };
  public relatedInformation?: DiagnosticRelatedInformation[];
  public tags?: number[];
  public data?: unknown;

  constructor(
    public readonly range: Range,
    public readonly message: string,
    public readonly severity: number = DiagnosticSeverity.Error,
  ) {}
}

export class Uri {
  public readonly scheme: string;
  public readonly path: string;
  public readonly fsPath: string;

  private constructor(scheme: string, p: string) {
    this.scheme = scheme;
    this.path   = p;
    this.fsPath = p;
  }

  static file(fsPath: string): Uri { return new Uri("file", fsPath); }
  static parse(value: string): Uri {
    const idx = value.indexOf(":");
    return new Uri(value.slice(0, idx), value.slice(idx + 1));
  }

  toString(): string { return `${this.scheme}:${this.path}`; }
  with(change: { scheme?: string; path?: string }): Uri {
    return new Uri(change.scheme ?? this.scheme, change.path ?? this.path);
  }
}

export class WorkspaceEdit {
  private _inserts: Array<{ uri: Uri; position: Position; text: string }> = [];
  private _replacements: Array<{ uri: Uri; range: Range; text: string }> = [];

  insert(uri: Uri, position: Position, text: string): void {
    this._inserts.push({ uri, position, text });
  }
  replace(uri: Uri, range: Range, text: string): void {
    this._replacements.push({ uri, range, text });
  }
  getInserts()      { return this._inserts; }
  getReplacements() { return this._replacements; }
}

export class CodeAction {
  public diagnostics?: Diagnostic[];
  public isPreferred?: boolean;
  public edit?: WorkspaceEdit;
  public command?: { command: string; title: string; arguments?: unknown[] };

  constructor(
    public readonly title: string,
    public readonly kind?: { value: string },
  ) {}
}

export class MarkdownString {
  public value: string;
  public isTrusted?: boolean;
  public supportHtml?: boolean;
  public supportThemeIcons?: boolean;

  constructor(value = "") {
    this.value = value;
  }

  appendMarkdown(text: string): this {
    this.value += text;
    return this;
  }

  appendText(text: string): this {
    // Escape markdown special characters — mirrors the real VS Code behaviour.
    this.value += text.replace(/[\\`*_{}[\]()#+\-.!|]/g, "\\$&");
    return this;
  }

  appendCodeblock(code: string, language = ""): this {
    this.value += `\`\`\`${language}\n${code}\n\`\`\`\n`;
    return this;
  }
}

export class Hover {
  constructor(
    public readonly contents: MarkdownString | MarkdownString[],
    public readonly range?: Range,
  ) {}
}

export class CodeLens {
  public command?: {
    title: string;
    command: string;
    arguments?: unknown[];
    tooltip?: string;
  };
  public isResolved = true;

  constructor(
    public readonly range: Range,
    command?: {
      title: string;
      command: string;
      arguments?: unknown[];
      tooltip?: string;
    },
  ) {
    this.command = command;
  }
}

/**
 * EventEmitter — real implementation so onDidChangeCodeLenses works correctly
 * and tests can verify that refresh() fires the event.
 */
export class EventEmitter<T = void> {
  private _listeners: Array<(e: T) => unknown> = [];

  get event(): (listener: (e: T) => unknown, thisArg?: unknown) => { dispose(): void } {
    return (listener) => {
      this._listeners.push(listener);
      return {
        dispose: () => {
          this._listeners = this._listeners.filter((l) => l !== listener);
        },
      };
    };
  }

  fire(data: T): void {
    for (const l of this._listeners) l(data);
  }

  dispose(): void {
    this._listeners = [];
  }
}

// Convenience alias — VS Code's Event<T> type is just a function signature.
export type Event<T> = (listener: (e: T) => unknown) => { dispose(): void };

// ---------------------------------------------------------------------------
// FakeWebviewPanel — returned by window.createWebviewPanel in tests.
// Tests can inspect `messageHandlers` / `disposeHandlers` or call
// `simulateMessage()` to drive the panel's onDidReceiveMessage callbacks.
// ---------------------------------------------------------------------------

export class FakeWebviewPanel {
  /** Handlers registered via webview.onDidReceiveMessage — inspectable in tests. */
  readonly messageHandlers: Array<(msg: unknown) => void> = [];
  /** Handlers registered via panel.onDidDispose — inspectable in tests. */
  readonly disposeHandlers: Array<() => void> = [];

  webview = {
    html: "",
    postMessage: jest.fn().mockResolvedValue(true),
    cspSource: "vscode-webview:",
    asWebviewUri: jest.fn((uri: Uri) => uri),
    // onDidReceiveMessage and onDidDispose are assigned in the constructor
    // so they can close over `this`.
    onDidReceiveMessage: jest.fn() as jest.MockedFunction<
      (
        handler: (msg: unknown) => void,
        thisArg?: unknown,
        disposables?: Array<{ dispose(): void }>,
      ) => { dispose(): void }
    >,
  };

  onDidDispose: jest.MockedFunction<
    (
      handler: () => void,
      thisArg?: unknown,
      disposables?: Array<{ dispose(): void }>,
    ) => { dispose(): void }
  > = jest.fn();

  reveal = jest.fn();
  dispose = jest.fn();

  constructor(
    public readonly viewType: string,
    public title: string,
  ) {
    // Capture the instance so closures work correctly.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const panel = this;

    this.webview.onDidReceiveMessage = jest.fn(
      (
        handler: (msg: unknown) => void,
        _thisArg?: unknown,
        disposables?: Array<{ dispose(): void }>,
      ) => {
        panel.messageHandlers.push(handler);
        const d = { dispose: jest.fn() };
        if (disposables) disposables.push(d);
        return d;
      },
    ) as typeof this.webview.onDidReceiveMessage;

    this.onDidDispose = jest.fn(
      (
        handler: () => void,
        _thisArg?: unknown,
        disposables?: Array<{ dispose(): void }>,
      ) => {
        panel.disposeHandlers.push(handler);
        const d = { dispose: jest.fn() };
        if (disposables) disposables.push(d);
        return d;
      },
    ) as typeof this.onDidDispose;

    this.dispose = jest.fn(() => {
      for (const h of panel.disposeHandlers) h();
    });
  }

  /**
   * Test helper: simulate a message posted from the webview to the extension.
   * Calls all handlers registered via `webview.onDidReceiveMessage`.
   */
  simulateMessage(message: unknown): void {
    for (const h of this.messageHandlers) h(message);
  }
}

// ---------------------------------------------------------------------------
// Fake CancellationToken returned by the default withProgress implementation.
// Tests that need to trigger cancellation can call _cancel() on it.
// ---------------------------------------------------------------------------

export class FakeCancellationToken {
  private _listeners: Array<() => void> = [];
  isCancellationRequested = false;

  onCancellationRequested(listener: () => void): { dispose(): void } {
    this._listeners.push(listener);
    return { dispose: () => { this._listeners = this._listeners.filter(l => l !== listener); } };
  }

  /** Test helper: simulate the user clicking ✕ on the progress notification. */
  _cancel(): void {
    this.isCancellationRequested = true;
    for (const l of this._listeners) l();
  }
}

// ---------------------------------------------------------------------------
// VS Code API namespace mocks
// ---------------------------------------------------------------------------

export const languages = {
  createDiagnosticCollection: jest.fn((_name?: string) => ({
    name: _name ?? "",
    set:     jest.fn(),
    delete:  jest.fn(),
    clear:   jest.fn(),
    forEach: jest.fn(),
    get:     jest.fn(),
    has:     jest.fn(),
    dispose: jest.fn(),
    [Symbol.iterator]: jest.fn(),
  })),
  registerCodeActionsProvider: jest.fn((_selector: unknown, _provider: unknown) => ({
    dispose: jest.fn(),
  })),
  registerCodeLensProvider: jest.fn((_selector: unknown, _provider: unknown) => ({
    dispose: jest.fn(),
  })),
  registerHoverProvider: jest.fn(() => ({ dispose: jest.fn() })),
};

/**
 * window.withProgress — default implementation actually executes the task
 * callback so orchestrator logic runs synchronously in unit tests.
 *
 * Tests that need to pause mid-run (double-trigger test) should override
 * this with jest.fn().mockImplementation(async (_opts, task) => { ... }).
 */
export const window = {
  showInformationMessage: jest.fn(),
  showWarningMessage:     jest.fn(),
  showErrorMessage:       jest.fn(),
  showQuickPick:          jest.fn(),

  createWebviewPanel: jest.fn(
    (viewType: string, title: string, _column: unknown, _options?: unknown) =>
      new FakeWebviewPanel(viewType, title),
  ),

  withProgress: jest.fn(async (
    _options: unknown,
    task: (
      progress: { report(v: unknown): void },
      token: FakeCancellationToken,
    ) => Thenable<unknown>,
  ) => {
    const fakeProgress = { report: jest.fn() };
    const fakeToken    = new FakeCancellationToken();
    return task(fakeProgress, fakeToken);
  }),

  createOutputChannel: jest.fn(() => ({
    appendLine: jest.fn(),
    show:       jest.fn(),
    dispose:    jest.fn(),
  })),
};

export const workspace = {
  getConfiguration: jest.fn((_section?: string) => ({
    get: jest.fn((_key: string) => undefined),
  })),
  openTextDocument: jest.fn(),
  textDocuments: [] as unknown[],
};

export const commands = {
  registerCommand: jest.fn((_id: string, _handler: unknown) => ({ dispose: jest.fn() })),
  executeCommand:  jest.fn(),
};

// ---------------------------------------------------------------------------
// Interface tags (TypeScript only — no runtime value needed)
// ---------------------------------------------------------------------------

export interface DiagnosticCollection {
  readonly name: string;
  set(uri: Uri, diagnostics: readonly Diagnostic[] | undefined): void;
  delete(uri: Uri): void;
  clear(): void;
  dispose(): void;
}

export interface TextDocument {
  readonly uri: Uri;
  readonly languageId: string;
  readonly lineCount: number;
  lineAt(line: number): { text: string };
}

export interface CodeActionContext {
  readonly diagnostics: readonly Diagnostic[];
  readonly only?: { value: string };
  readonly triggerKind?: number;
}

export interface CancellationToken {
  readonly isCancellationRequested: boolean;
  onCancellationRequested(listener: () => void): { dispose(): void };
}

export interface ExtensionContext {
  subscriptions: { dispose(): unknown }[];
  extensionPath: string;
}

export interface QuickPickItem {
  label: string;
  description?: string;
  detail?: string;
  picked?: boolean;
}

export interface HoverProvider {
  provideHover(
    document: TextDocument,
    position: Position,
  ): Hover | undefined | Promise<Hover | undefined>;
}

export interface TextDocumentShowOptions {
  selection?: Range;
  viewColumn?: number;
  preserveFocus?: boolean;
  preview?: boolean;
}
