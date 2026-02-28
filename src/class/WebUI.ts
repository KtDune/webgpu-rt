import { vec3 } from 'gl-matrix';

export class WebUI {
    private pbrToggle: HTMLInputElement;
    private lightX: HTMLInputElement;
    private lightY: HTMLInputElement;
    private lightZ: HTMLInputElement;
    private lightXValue: HTMLElement;
    private lightYValue: HTMLElement;
    private lightZValue: HTMLElement;

    private _lightPos: vec3 = vec3.fromValues(1, 1, 1);
    private _usePBR: boolean = true;
    private needsUpdate: boolean = true;

    constructor() {
        this.pbrToggle = document.getElementById('pbr-toggle') as HTMLInputElement;
        this.lightX = document.getElementById('light-x') as HTMLInputElement;
        this.lightY = document.getElementById('light-y') as HTMLInputElement;
        this.lightZ = document.getElementById('light-z') as HTMLInputElement;
        this.lightXValue = document.getElementById('light-x-value') as HTMLElement;
        this.lightYValue = document.getElementById('light-y-value') as HTMLElement;
        this.lightZValue = document.getElementById('light-z-value') as HTMLElement;

        // Initialize _lightPos from slider HTML values
        this._lightPos = vec3.fromValues(
            parseFloat(this.lightX.value),
            parseFloat(this.lightY.value),
            parseFloat(this.lightZ.value)
        );

        this.updateDisplayValues();
        this.setupEventListeners();
    }

    private setupEventListeners(): void {
        this.pbrToggle.addEventListener('change', () => {
            this._usePBR = this.pbrToggle.checked;
            this.needsUpdate = true;
        });

        this.lightX.addEventListener('input', () => this.handleSliderChange());
        this.lightY.addEventListener('input', () => this.handleSliderChange());
        this.lightZ.addEventListener('input', () => this.handleSliderChange());
    }

    private handleSliderChange(): void {
        this.updateDisplayValues();

        vec3.set(
            this._lightPos,
            parseFloat(this.lightX.value),
            parseFloat(this.lightY.value),
            parseFloat(this.lightZ.value)
        );

        this.needsUpdate = true;
    }

    private updateDisplayValues(): void {
        this.lightXValue.textContent = this.lightX.value;
        this.lightYValue.textContent = this.lightY.value;
        this.lightZValue.textContent = this.lightZ.value;
    }

    get lightPosition(): vec3 {
        return this._lightPos;
    }

    get usePBR(): number {
        return this._usePBR ? 1 : 0;
    }

    get needsUniformUpdate(): boolean {
        return this.needsUpdate;
    }

    consumeUpdate(): { usePBR: number, lightPos: vec3 } | null {
        if (this.needsUpdate) {
            this.needsUpdate = false;
            return {
                usePBR: this._usePBR ? 1 : 0,
                lightPos: this._lightPos
            };
        }
        return null;
    }

    reset(): void {
        this.lightX.value = '0';
        this.lightY.value = '5';
        this.lightZ.value = '10';
        this.pbrToggle.checked = true;
        this.handleSliderChange();
    }
}