// largely copied from efficiency nodes!
import { app } from "../../scripts/app.js";

let origProps = {};

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

const doesInputWithNameExist = (node, name) => {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
};

const HIDDEN_TAG = "gendatahide";
// Toggle Widget + change size
function toggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;

    // Store the original properties of the widget if not already stored
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }

    const origSize = node.size;

    // special behavior for textareas -- before we update type
    if (widget.inputEl?.localName === 'textarea') {
        widget.inputEl.className = show ? 'comfy-multiline-input' : 'comfy-multiline-input comfy-multiline-input-hidden';
        // widget.inputEl.style.display = show ? 'absolute' : 'none'; 
        // console.log('set style.display to', widget.inputEl.style.display);
    }
    // Set the widget type and computeSize based on the show flag
    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    // Recursively handle linked widgets if they exist
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));

    // console.log('widget', widget);

    // Calculate the new height for the node based on its computeSize method
    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

// New function to handle widget visibility based on input_mode
function handleWidgetsVisibility(node, widgetCount, maxCount, inputModeValue) {
    const allNames = [ "ckpt", "vae", "clipskip", "gendata" ];
    const nodeVisibilityMap = {
        "Checkpoint Selector Stacker 👩‍💻": {
            "checkpoint only": [ "ckpt" ],
            "checkpoint + vae": [ "ckpt", "vae" ],
            "checkpoint + vae + clip skip": [ "ckpt", "vae", "clipskip" ],
        },
        "GenData Stacker 👩‍💻": [ "gendata" ],
    };

    const inputModeVisibilityMap = nodeVisibilityMap[node.comfyClass];
    if (!inputModeVisibilityMap) return;

    const widgetMap = Array.isArray(inputModeVisibilityMap) ? inputModeVisibilityMap : inputModeVisibilityMap[inputModeValue];
    if (!widgetMap) return;

    for (let i=0; i<maxCount; i++) {
        for (let n of allNames) {
            const widget = findWidgetByName(node, `${n}_${i}`);
            const showWidget = widgetMap.includes(n) && i <= widgetCount;
            if (widget) {
                toggleWidget(node, widget, showWidget);
            }
        }
    }
}

// Create a map of node titles to their respective widget handlers
const nodeWidgetHandlers = {
    "Checkpoint Selector Stacker 👩‍💻": {
        'ckpt_count': handleCheckpointSelectorStackerCount,
        'input_mode': handleCheckpointSelectorStackerInputMode,
    },
    "GenData Stacker 👩‍💻": {
        'gendata_count': handleGenDataStacker,
    },
};

function handleGenDataStacker(node, widget) {
    handleWidgetsVisibility(node, widget.value, 50);
}

function handleCheckpointSelectorStackerCount(node, widget) {
    const inputModeValue = findWidgetByName(node, "input_mode").value;
    handleWidgetsVisibility(node, widget.value, 50, inputModeValue);
}

function handleCheckpointSelectorStackerInputMode(node, widget) {
    const ckpt_count = findWidgetByName(node, "ckpt_count").value;
    handleWidgetsVisibility(node, ckpt_count, 50, widget.value);
}

// In the main function where widgetLogic is called
function widgetLogic(node, widget) {
    // Retrieve the handler for the current node title and widget name
    const handler = nodeWidgetHandlers[node.comfyClass]?.[widget.name];
    if (handler) {
        handler(node, widget);
    }
}

app.registerExtension({
    name: "gendata.widgethider",
    init() {
        // inject CSS - necessary to be able to truly hide <textarea> multiline elements
        window.setTimeout(() => {
            let link = document.createElement("link");
            link.rel = "stylesheet";
            link.type = "text/css";
            link.href = "extensions/ComfyUI-GenData-Pack/gendata.css";
            document.head.appendChild(link);
        }, 100);
    },
    nodeCreated(node) {
        for (const w of node.widgets || []) {
            let widgetValue = w.value;

            // Store the original descriptor if it exists
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value');

            widgetLogic(node, w);

            Object.defineProperty(w, 'value', {
                get() {
                    // If there's an original getter, use it. Otherwise, return widgetValue.
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {

                    // If there's an original setter, use it. Otherwise, set widgetValue.
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else {
                        widgetValue = newVal;
                    }

                    widgetLogic(node, w);
                }
            });
        }
    }
});

