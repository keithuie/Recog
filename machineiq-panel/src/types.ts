export interface MachineIQOptions {
    thresholdGreen: number;
    thresholdYellow: number;
    showDetails: boolean;
}

// Default values for the options
export const defaults: MachineIQOptions = {
    thresholdGreen: 80,
    thresholdYellow: 50,
    showDetails: true,
};
