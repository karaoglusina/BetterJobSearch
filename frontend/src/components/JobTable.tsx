import { useMemo, useRef, useEffect, useCallback } from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { GridApi, SelectionChangedEvent, RowClickedEvent, ColDef } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import type { Job } from '../api/client';
import type { JobLabels } from '../hooks/useLabels';

interface JobTableProps {
  jobs: Job[];
  onJobSelect?: (jobId: string) => void;
  onSelectionChange?: (ids: Set<string>) => void;
  selectedJobIds?: Set<string>;
  jobLabels?: JobLabels;
}

export default function JobTable({
  jobs,
  onJobSelect,
  onSelectionChange,
  selectedJobIds,
  jobLabels = {},
}: JobTableProps) {
  const gridRef = useRef<AgGridReact>(null);
  const gridApiRef = useRef<GridApi | null>(null);

  // Enrich jobs with labels and selection status
  const enrichedJobs = useMemo(() => {
    return jobs.map(job => ({
      ...job,
      labels: (jobLabels[job.job_id] || []).join(', '),
      isSelected: selectedJobIds?.has(job.job_id) ?? false,
    }));
  }, [jobs, jobLabels, selectedJobIds]);

  const columnDefs = useMemo((): ColDef[] => [
    {
      headerCheckboxSelection: true,
      checkboxSelection: true,
      width: 50,
      maxWidth: 50,
      suppressSizeToFit: true,
    },
    {
      field: 'isSelected',
      headerName: '★',
      width: 45,
      maxWidth: 45,
      cellRenderer: (params: any) => params.value ? '★' : '',
      cellStyle: { color: '#f59e0b', textAlign: 'center' },
      comparator: (a: boolean, b: boolean) => (b ? 1 : 0) - (a ? 1 : 0), // Selected items first
    },
    { field: 'title', headerName: 'Title', flex: 2, minWidth: 180 },
    { field: 'company', headerName: 'Company', flex: 1, minWidth: 120 },
    { field: 'location', headerName: 'Location', flex: 1, minWidth: 100 },
    { field: 'labels', headerName: 'Labels', width: 100, cellStyle: { color: '#6366f1' } },
    { field: 'days_old', headerName: 'Days', width: 60, type: 'numericColumn' },
    { field: 'contract_type', headerName: 'Type', width: 80 },
    { field: 'work_type', headerName: 'Work', width: 80 },
    { field: 'language', headerName: 'Lang', width: 50 },
  ], []);

  const defaultColDef = useMemo(() => ({
    sortable: true,
    filter: true,
    resizable: true,
  }), []);

  // Handle selection changes
  const handleSelectionChanged = useCallback((event: SelectionChangedEvent) => {
    if (!onSelectionChange) return;
    const selected = event.api.getSelectedRows();
    const ids = new Set(selected.map((row: any) => row.job_id));
    onSelectionChange(ids);
  }, [onSelectionChange]);

  // Handle row click (for single selection / detail view)
  const handleRowClicked = useCallback((event: RowClickedEvent) => {
    if (event.data && onJobSelect) {
      onJobSelect(event.data.job_id);
    }
  }, [onJobSelect]);

  // Sync external selection with grid
  useEffect(() => {
    const api = gridApiRef.current;
    if (!api) return;

    // Use setTimeout to ensure this runs after grid is fully rendered
    const timeoutId = setTimeout(() => {
      if (!selectedJobIds || selectedJobIds.size === 0) {
        api.deselectAll();
        return;
      }

      // Prevent infinite loops by checking if selection actually changed
      const currentSelection = new Set(api.getSelectedRows().map((r: any) => r.job_id));
      const isSame = currentSelection.size === selectedJobIds.size &&
        Array.from(selectedJobIds).every(id => currentSelection.has(id));

      if (!isSame) {
        api.deselectAll();
        api.forEachNode(node => {
          if (node.data && selectedJobIds.has(node.data.job_id)) {
            node.setSelected(true, false, 'api'); // true = selected, false = don't clear others, 'api' = source
          }
        });
      }
    }, 0);

    return () => clearTimeout(timeoutId);
  }, [selectedJobIds, enrichedJobs]); // Also depend on enrichedJobs to re-sync when data changes

  return (
    <div className="ag-theme-alpine-dark" style={{ width: '100%', height: '100%' }}>
      <AgGridReact
        ref={gridRef}
        rowData={enrichedJobs}
        columnDefs={columnDefs}
        defaultColDef={defaultColDef}
        rowSelection="multiple"
        suppressRowClickSelection={true}
        getRowId={(params) => params.data.job_id}
        onGridReady={(params) => {
          gridApiRef.current = params.api;
        }}
        onSelectionChanged={handleSelectionChanged}
        onRowClicked={handleRowClicked}
        animateRows={true}
        pagination={true}
        paginationPageSize={25}
      />
    </div>
  );
}
