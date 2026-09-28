import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const remoteNodes = new Set([
  "TextEncodeQwenImage21Remote",
  "TextEncodeQwenImageEditRemote",
  "QwenImage21Remote",
  "MiniMaxH3ImageToVideoRemote",
]);

app.registerExtension({
  name: "remote-te.status",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!remoteNodes.has(nodeData.name)) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const widget = ComfyWidgets.STRING(
        this,
        "remote_te_status",
        ["STRING", { multiline: true }],
        app,
      ).widget;
      widget.options.serialize = false;
      widget.serialize = false;
      widget.value = "Remote encoder: waiting for execution";
      if (widget.inputEl) widget.inputEl.readOnly = true;
      this.remoteTEStatus = widget;
      return result;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = executed?.apply(this, arguments);
      if (this.remoteTEStatus && Array.isArray(message?.text)) {
        this.remoteTEStatus.value = message.text.join("\n");
        this.setDirtyCanvas(true, true);
      }
      return result;
    };
  },
});
