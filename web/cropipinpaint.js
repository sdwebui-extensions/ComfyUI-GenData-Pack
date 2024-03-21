import { app } from "../../scripts/app.js";

const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";
const RANDOMIZE_BUTTON_LABEL = "🎲 Randomize Each Time";
const NEW_FIXED_RANDOM_BUTTON_LABEL = "🎲 New Fixed Random";

const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];
const HIDDEN_TAG = "gendatahide";
const MODE_NORMAL = LiteGraph.ALWAYS;
const MODE_MUTED = LiteGraph.NEVER;

class CropIpInpaint {
    constructor(node) {
        // seed handling
        this.lastSeed = undefined;
        this.serializedCtx = {};
        this.lastSeedValue = null;
        this.node = node;
        this.graph = null;

        this.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff = null;
        this.processingQueue = false;
        this.initializeComfyUIHooks();

        const stagehide_crop = ["use_ip_adapter", "ip_weight", "ip_noise", "ip_weight_type", "steps", "cfg", "sampler_name", "scheduler", "denoise", "overlay_blur_amount", "seed", RANDOMIZE_BUTTON_LABEL, NEW_FIXED_RANDOM_BUTTON_LABEL];
        const always_hidden = ['image'];
        this.node.widgets.forEach(w => {
            w.visibleInStage = [];
            if (w.name === 'output_stage') {
                // obviously don't hide the visibility controller
                w.visibleInStage = ['Crop', 'Render', 'Final'];
            } else if (!always_hidden.includes(w.name)) {
                if (!stagehide_crop.includes(w.name)) {
                    w.visibleInStage = ['Crop'];
                }
                if (w.name !== "overlay_blur_amount") {
                    w.visibleInStage = [...w.visibleInStage, 'Render'];
                }
                // everything not always hidden is visible in Final
                w.visibleInStage = [...w.visibleInStage, 'Final'];
            }
        });

        this.node.constructor.exposedActions = ["Randomize Each Time", "Use Last Queued Seed"];
        const handleAction = this.node.handleAction;
        this.node.handleAction = async (action) => {
            handleAction && handleAction.call(this.node, action);
            if (action === "Randomize Each Time") {
                this.seedWidget.value = SPECIAL_SEED_RANDOM;
            }
            else if (action === "Use Last Queued Seed") {
                this.seedWidget.value = this.lastSeed != null ? this.lastSeed : this.seedWidget.value;
                this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                this.lastSeedButton.disabled = true;
            }
        };
        this.node.properties = this.node.properties || {};
        for (const [i, w] of this.node.widgets.entries()) {
            if (w.name === "seed") {
                this.seedWidget = w;
                this.seedWidget.value = SPECIAL_SEED_RANDOM;
            }
            else if (w.name === "control_after_generate") {
                this.node.widgets.splice(i, 1);
            }
        }
        if (!this.seedWidget) {
            throw new Error("Something's wrong; expected seed widget");
        }
        const randMax = Math.min(1125899906842624, this.seedWidget.options.max);
        const randMin = Math.max(0, this.seedWidget.options.min);
        const randomRange = (randMax - Math.max(0, randMin)) / (this.seedWidget.options.step / 10);
        this.randomizeSeedButton = this.node.addWidget("button", RANDOMIZE_BUTTON_LABEL, null, () => {
            this.seedWidget.value = SPECIAL_SEED_RANDOM;
        }, { serialize: false });
        this.newFixedRandomButton = this.node.addWidget("button", NEW_FIXED_RANDOM_BUTTON_LABEL, null, () => {
            this.seedWidget.value =
                Math.floor(Math.random() * randomRange) * (this.seedWidget.options.step / 10) + randMin;
        }, { serialize: false });
        this.lastSeedButton = this.node.addWidget("button", LAST_SEED_BUTTON_LABEL, null, () => {
            this.seedWidget.value = this.lastSeed != null ? this.lastSeed : this.seedWidget.value;
            this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
            this.lastSeedButton.disabled = true;
        }, { width: 50, serialize: false });
        this.lastSeedButton.disabled = true;

        this.randomizeSeedButton.visibleInStage = ['Render', 'Final'];
        this.newFixedRandomButton.visibleInStage = ['Render', 'Final'];
        this.lastSeedButton.visibleInStage = ['Render', 'Final']; // we hide this one in a different way

        this.seedWidget.serializeValue = async (node, index) => {
            if (!this.graph && app.graph) {
                this.graph = app.graph;
                this.initializeGraphAndCanvasHooks();
            }

            const inputSeed = this.seedWidget.value;
            if (!this.processingQueue) {
                return inputSeed;
            }
            this.serializedCtx = {
                inputSeed: this.seedWidget.value,
            };
            if (SPECIAL_SEEDS.includes(this.serializedCtx.inputSeed)) {
                if (typeof this.lastSeed === "number" && !SPECIAL_SEEDS.includes(this.lastSeed)) {
                    if (inputSeed === SPECIAL_SEED_INCREMENT) {
                        this.serializedCtx.seedUsed = this.lastSeed + 1;
                    }
                    else if (inputSeed === SPECIAL_SEED_DECREMENT) {
                        this.serializedCtx.seedUsed = this.lastSeed - 1;
                    }
                }
                if (!this.serializedCtx.seedUsed || SPECIAL_SEEDS.includes(this.serializedCtx.seedUsed)) {
                    this.serializedCtx.seedUsed =
                        Math.floor(Math.random() * randomRange) * (this.seedWidget.options.step / 10) + randMin;
                }
            }
            else {
                this.serializedCtx.seedUsed = this.seedWidget.value;
            }
            if (this.graph.onSerialize) {
                const n = this.getNodeFromInitialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff(node);
                if (n) {
                    n.widgets_values[index] = this.serializedCtx.seedUsed;
                }
                else {
                    console.warn('No serialized node found in workflow. May be attributed to '
                        + 'https://github.com/comfyanonymous/ComfyUI/issues/2193');
                }
            }
            this.seedWidget.value = this.serializedCtx.seedUsed;
            this.lastSeed = this.serializedCtx.seedUsed;
            if (SPECIAL_SEEDS.includes(this.serializedCtx.inputSeed)) {
                this.lastSeedButton.name = `♻️ ${this.serializedCtx.seedUsed}`;
                this.lastSeedButton.disabled = false;
                if (this.lastSeedValue) {
                    this.lastSeedValue.value = `Last Seed: ${this.serializedCtx.seedUsed}`;
                }
            }
            else {
                this.lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                this.lastSeedButton.disabled = true;
            }
            return this.serializedCtx.seedUsed;
        };
        this.seedWidget.afterQueued = () => {
            if (this.serializedCtx.inputSeed) {
                this.seedWidget.value = this.serializedCtx.inputSeed;
            }
            this.serializedCtx = {};
        };

        this.handleVisibility();
        const _this = this;

        // piggyback on get/set to listen for changes to stage
        const wStage = this.findWidget('output_stage');
        let wStage_value = wStage.value;
        const wStage_originalDescriptor = Object.getOwnPropertyDescriptor(wStage, 'value');
        if (wStage) {
            Object.defineProperty(wStage, 'value', {
                get() {
                    return wStage_originalDescriptor && wStage_originalDescriptor.get ? wStage_originalDescriptor.get.call(wStage) : wStage_value;
                },
                set(newVal) {
                    if (wStage_originalDescriptor && wStage_originalDescriptor.set) {
                        wStage_originalDescriptor.set.call(wStage, newVal);
                    } else {
                        wStage_value = newVal;
                    }

                    _this.handleVisibility.call(_this);
                },
            });
        }
    }
    initializeComfyUIHooks() {
        const queuePrompt = app.queuePrompt;
        const gthis = this;
        app.queuePrompt = async function () {
            gthis.processingQueue = true;
            try {
                await queuePrompt.apply(app, [...arguments]);
            }
            finally {
                gthis.processingQueue = false;
            }
        }
    };
    async initializeGraphAndCanvasHooks() {
        const graph = this.graph;
        const onSerialize = graph.onSerialize;
        graph.onSerialize = (data) => {
            this.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff = data;
            onSerialize === null || onSerialize === void 0 ? void 0 : onSerialize.call(graph, data);
        };
    };
    getNodeFromInitialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff(node) {
        var _a, _b, _c;
        return ((_c = (_b = (_a = this.initialGraphToPromptSerializedWorkflowBecauseComfyUIBrokeStuff) === null || _a === void 0 ? void 0 : _a.nodes) === null || _b === void 0 ? void 0 : _b.find((n) => n.id === node.id)) !== null && _c !== void 0 ? _c : null);
    };
    findWidget(name, includesChar = false) {
        if (includesChar) {
            return this.node.widgets.find(w => w.name.indexOf(name) > -1) || undefined;
        }
        return this.node.widgets.find(w => w.name === name) || undefined
    };
    doesInputWithNameExist = (name) => {
        return this.node.inputs ? this.node.inputs.some((input) => input.name === name) : false;
    };
    findOutput(name) {
        return this.node.outputs.find(o => o.name === name) || undefined;
    };
    setWidgetVisibility(widget, updateSize = false, show = false) {
        if (!widget || this.doesInputWithNameExist(widget.name)) return;

        // Store the original properties of the widget if not already stored
        if (!widget.origProps) {
            widget.origProps = { origType: widget.type, origComputeSize: widget.computeSize };
        }

        // special behavior for textareas -- before we update type
        if (widget.inputEl?.localName === 'textarea') {
            widget.inputEl.className = show ? 'comfy-multiline-input' : 'comfy-multiline-input comfy-multiline-input-hidden';
        }
        // Set the widget type and computeSize based on the show flag
        widget.type = show ? widget.origProps.origType : HIDDEN_TAG;
        widget.computeSize = show ? widget.origProps.origComputeSize : () => [0, -4];

        // Recursively handle linked widgets if they exist
        widget.linkedWidgets?.forEach(w => this.setWidgetVisibility(w, updateSize, show));

        // Calculate the new height for the node based on its computeSize method
        if (updateSize) {
            const newHeight = this.node.computeSize()[1];
            this.node.setSize([this.node.size[0], newHeight]);
        }
    };
    toggleOutputTarget(output_name, isNormal = () => true) {
        const outp = this.findOutput(output_name);
        if ((outp?.links || []).length > 0) {
            const graph = app.graph;
            outp.links.forEach(k => {
                const graphlink = graph.links[k];
                const targetnode = graph._nodes_by_id[graphlink.target_id];

                targetnode.mode = isNormal() ? MODE_NORMAL : MODE_MUTED;
            });
        }
    };
    handleVisibility() {
        // hide image name
        this.setWidgetVisibility(this.findWidget("image"), false, false);

        // stage based visibility
        const wStage = this.findWidget("output_stage");
        if (!wStage) return;

        const stage = wStage.value;
        this.node.widgets.forEach(w => {
            if (w.visibleInStage.includes(stage)) {
                this.setWidgetVisibility(w, false, true);
            } else {
                this.setWidgetVisibility(w, false, false);
            }
        });

        this.toggleOutputTarget('image_final', () => stage === 'Final');
        this.toggleOutputTarget('image_render', () => stage === 'Final' || stage === 'Render');
        this.toggleOutputTarget('latent_render', () => stage === 'Final' || stage === 'Render');

        // const image_final = this.findOutput('image_final');
        // if ((image_final?.links || []).length > 0) {
        //     const graph = app.graph;
        //     image_final.links.forEach(k => {
        //         const graphlink = graph.links[k];
        //         const targetnode = graph._nodes_by_id[graphlink.target_id];

        //         targetnode.mode = stage === "Final" ? MODE_NORMAL : MODE_MUTED;
        //     });
        // }

        // const image_render = this.findOutput('image_render');
        // if ((image_render?.links || []).length > 0) {
        //     const graph = app.graph;
        //     image_render.links.forEach(k => {
        //         const graphlink = graph.links[k];
        //         const targetnode = graph._nodes_by_id[graphlink.target_id];

        //         targetnode.mode = stage === "Final" || stage === "Render" ? MODE_NORMAL : MODE_MUTED;
        //     });
        // }
        app.graph.setDirtyCanvas(true, false); // fg, bg
    };
}
app.registerExtension({
    name: "GenData.Seed",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "Crop|IP|Inpaint|SDXL 👩‍💻" || nodeData.name === "Crop|IP|Inpaint 👩‍💻") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                this.cropIpInpaint = new CropIpInpaint(this);
            };
        }
    },
});
