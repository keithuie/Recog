import { PanelPlugin } from '@grafana/data';
import { MachineIQOptions, defaults } from './types';
import { MachineIQPanel } from './components/MachineIQPanel';

export const plugin = new PanelPlugin<MachineIQOptions>(MachineIQPanel).setPanelOptions((builder) => {
    return builder
        .addNumberInput({
            path: 'thresholdGreen',
            name: 'Green Threshold',
            description: 'Confidence level above which the bar turns green',
            defaultValue: defaults.thresholdGreen,
        })
        .addNumberInput({
            path: 'thresholdYellow',
            name: 'Yellow Threshold',
            description: 'Confidence level above which the bar turns yellow',
            defaultValue: defaults.thresholdYellow,
        })
        .addBooleanSwitch({
            path: 'showDetails',
            name: 'Show Percentage',
            description: 'Show the exact confidence percentage on the bar',
            defaultValue: defaults.showDetails,
        });
});
