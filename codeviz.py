import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple


class DataAnalyzer:
    def __init__(self):
        """Initialize the data analyzer"""
        self.df = None
        self.data_summary = None
        self.available_analyses = {
            'Scatter Plot': self.create_scatter_plot,
            'Line Plot': self.create_line_plot,
            'Bar Chart': self.create_bar_chart,
            'Histogram': self.create_histogram,
            'Box Plot': self.create_box_plot,
            'Summary Statistics': self.get_summary_statistics,
            'Correlation Analysis': self.create_correlation_heatmap,
            'Group Analysis': self.analyze_by_group
        }

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate the uploaded data"""
        try:
            if df.empty:
                return False, "The uploaded file contains no data."

            if len(df) < 2:
                return False, "The dataset must contain at least 2 rows."

            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return False, "The dataset must contain at least one numeric column."

            return True, "Data validation successful."

        except Exception as e:
            return False, f"Data validation failed: {str(e)}"

    def generate_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate a summary of the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': list(numeric_cols),
            'categorical_columns': list(categorical_cols),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
            'categorical_summary': {col: df[col].value_counts().to_dict() for col in categorical_cols}
        }
        return summary

    def load_data(self, file) -> Tuple[bool, str]:
        """Load and validate data from uploaded file"""
        try:
            if file.name.endswith('.csv'):
                self.df = pd.read_csv(file)
            else:
                self.df = pd.read_excel(file)

            is_valid, message = self.validate_data(self.df)
            if not is_valid:
                return False, message

            self.data_summary = self.generate_data_summary(self.df)
            return True, "Data loaded successfully"

        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def create_scatter_plot(self, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
        """Create a scatter plot"""
        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f'{y_col} vs {x_col}'
        )
        return fig

    def create_line_plot(self, x_col: str, y_col: str, group_col: str = None) -> go.Figure:
        """Create a line plot"""
        fig = px.line(
            self.df,
            x=x_col,
            y=y_col,
            color=group_col,
            title=f'{y_col} over {x_col}'
        )
        return fig

    def create_bar_chart(self, x_col: str, y_col: str, group_col: str = None) -> go.Figure:
        """Create a bar chart"""
        fig = px.bar(
            self.df,
            x=x_col,
            y=y_col,
            color=group_col,
            title=f'{y_col} by {x_col}'
        )
        return fig

    def create_histogram(self, column: str, bins: int = 30) -> go.Figure:
        """Create a histogram"""
        fig = px.histogram(
            self.df,
            x=column,
            nbins=bins,
            title=f'Distribution of {column}'
        )
        return fig

    def create_box_plot(self, numeric_col: str, category_col: str = None) -> go.Figure:
        """Create a box plot"""
        fig = px.box(
            self.df,
            x=category_col,
            y=numeric_col,
            title=f'Box Plot of {numeric_col}'
        )
        return fig

    def create_correlation_heatmap(self, columns: list = None) -> go.Figure:
        """Create a correlation heatmap"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        corr_matrix = self.df[columns].corr()
        fig = px.imshow(
            corr_matrix,
            title='Correlation Heatmap',
            aspect='auto'
        )
        return fig

    def get_summary_statistics(self, columns: list = None) -> pd.DataFrame:
        """Get summary statistics for specified columns"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        return self.df[columns].describe()

    def analyze_by_group(self, group_col: str, value_col: str) -> pd.DataFrame:
        """Analyze a value column grouped by a categorical column"""
        return self.df.groupby(group_col)[value_col].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)


def main():
    st.set_page_config(page_title="Data Analysis Tool", layout="wide")

    st.title("Interactive Data Analysis Tool")

    # Initialize the analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()

    # File upload
    uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        # Load and validate data
        success, message = st.session_state.analyzer.load_data(uploaded_file)

        if not success:
            st.error(message)
            return

        # Display data summary
        with st.expander("Dataset Summary"):
            st.json(st.session_state.analyzer.data_summary)

        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.analyzer.df.head())

        # Analysis selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            options=list(st.session_state.analyzer.available_analyses.keys())
        )

        # Analysis parameters
        with st.form("analysis_params"):
            if analysis_type in ['Scatter Plot', 'Line Plot', 'Bar Chart']:
                x_col = st.selectbox("Select X-axis column", st.session_state.analyzer.df.columns)
                y_col = st.selectbox("Select Y-axis column", st.session_state.analyzer.df.columns)
                color_col = st.selectbox("Select grouping column (optional)",
                                         ['None'] + list(st.session_state.analyzer.df.columns))

            elif analysis_type in ['Histogram', 'Box Plot']:
                x_col = st.selectbox("Select column to analyze",
                                     st.session_state.analyzer.df.select_dtypes(include=[np.number]).columns)
                y_col = None
                color_col = None

            elif analysis_type == 'Correlation Analysis':
                x_col = st.multiselect(
                    "Select columns for correlation analysis",
                    st.session_state.analyzer.df.select_dtypes(include=[np.number]).columns
                )
                y_col = None
                color_col = None

            elif analysis_type == 'Group Analysis':
                x_col = st.selectbox("Select grouping column",
                                     st.session_state.analyzer.df.select_dtypes(exclude=[np.number]).columns)
                y_col = st.selectbox("Select value column",
                                     st.session_state.analyzer.df.select_dtypes(include=[np.number]).columns)
                color_col = None

            submitted = st.form_submit_button("Generate Analysis")

        if submitted:
            try:
                # Get the appropriate analysis function
                analysis_func = st.session_state.analyzer.available_analyses[analysis_type]

                # Execute analysis and display results
                if analysis_type in ['Scatter Plot', 'Line Plot', 'Bar Chart']:
                    fig = analysis_func(x_col, y_col, None if color_col == 'None' else color_col)
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis_type in ['Histogram', 'Box Plot']:
                    fig = analysis_func(x_col)
                    st.plotly_chart(fig, use_container_width=True)

                elif analysis_type == 'Correlation Analysis':
                    if len(x_col) < 2:
                        st.error("Please select at least 2 columns for correlation analysis")
                    else:
                        fig = analysis_func(x_col)
                        st.plotly_chart(fig, use_container_width=True)

                elif analysis_type in ['Summary Statistics', 'Group Analysis']:
                    result = analysis_func(x_col, y_col)
                    st.dataframe(result)

            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")


if __name__ == "__main__":
    main()
