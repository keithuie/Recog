import React from 'react';
import { PanelProps } from '@grafana/data';
import { MachineIQOptions } from '../types';
import { useTheme2 } from '@grafana/ui';

interface Props extends PanelProps<MachineIQOptions> { }

export const MachineIQPanel: React.FC<Props> = ({ options, data, width, height }) => {
    const theme = useTheme2();

    // 1. Data Extraction
    // We expect the query to return fields: "confidence" (number) and "pattern" (string)
    let confidence = 0;
    let patternName = "Scanning...";

    if (data.series.length > 0) {
        const series = data.series[0];

        // Attempt to find fields by name or type
        const confidenceField = series.fields.find(f => f.name === 'confidence' || f.type === 'number');
        const patternField = series.fields.find(f => f.name === 'pattern' || f.type === 'string');

        if (confidenceField) {
            const vals = confidenceField.values;
            confidence = vals.get(vals.length - 1) || 0;
        }

        if (patternField) {
            const vals = patternField.values;
            patternName = vals.get(vals.length - 1) || "Unknown";
        }
    }

    // 2. Color Logic
    let barColor = theme.colors.error.main; // Red
    if (confidence >= options.thresholdGreen) {
        barColor = theme.colors.success.main; // Green
    } else if (confidence >= options.thresholdYellow) {
        barColor = theme.colors.warning.main; // Yellow
    }

    // 3. Geometry Calculation
    const padding = 20;
    const safeWidth = width - (padding * 2);
    const barHeight = 40;
    const barWidth = (Math.min(confidence, 100) / 100) * safeWidth;
    const centerY = height / 2;

    return (
        <div style={{ width, height, position: 'relative' }}>
            <svg width={width} height={height}>
                {/* Background Track */}
                <rect
                    x={padding}
                    y={centerY - (barHeight / 2)}
                    width={safeWidth}
                    height={barHeight}
                    fill={theme.colors.background.secondary}
                    rx={4}
                />

                {/* Confidence Bar */}
                <rect
                    x={padding}
                    y={centerY - (barHeight / 2)}
                    width={barWidth}
                    height={barHeight}
                    fill={barColor}
                    rx={4}
                    style={{ transition: 'width 0.5s ease-in-out, fill 0.5s ease-in-out' }}
                />

                {/* Pattern Name Label */}
                <text
                    x={width / 2}
                    y={centerY - 30}
                    textAnchor="middle"
                    fill={theme.colors.text.primary}
                    fontSize="18px"
                    fontWeight="bold"
                >
                    {patternName}
                </text>

                {/* Percentage Label */}
                {options.showDetails && (
                    <text
                        x={width / 2}
                        y={centerY + 6}
                        textAnchor="middle"
                        fill={theme.colors.text.primary} // Contrast text
                        fontSize="14px"
                        fontWeight="bold"
                        style={{ textShadow: '0px 1px 2px rgba(0,0,0,0.5)' }}
                    >
                        {confidence.toFixed(1)}%
                    </text>
                )}
            </svg>
        </div>
    );
};
