import React, { useMemo, useRef } from 'react';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import type { Job } from '../api/client';

interface JobTableProps {
  jobs: Job[];
  onJobSelect?: (jobId: string) => void;
  selectedJobId?: string | null;
}

export default function JobTable({ jobs, onJobSelect, selectedJobId }: JobTableProps) {
  const gridRef = useRef<AgGridReact>(null);

  const columnDefs = useMemo(() => [
    { field: 'title', headerName: 'Title', flex: 2, minWidth: 200 },
    { field: 'company', headerName: 'Company', flex: 1, minWidth: 150 },
    { field: 'location', headerName: 'Location', flex: 1, minWidth: 120 },
    { field: 'days_old', headerName: 'Days', width: 70, type: 'numericColumn' },
    { field: 'contract_type', headerName: 'Type', width: 100 },
    { field: 'work_type', headerName: 'Work', width: 100 },
  ], []);

  const defaultColDef = useMemo(() => ({
    sortable: true,
    filter: true,
    resizable: true,
  }), []);

  return (
    <div className="ag-theme-alpine-dark" style={{ width: '100%', height: '100%' }}>
      <AgGridReact
        ref={gridRef}
        rowData={jobs}
        columnDefs={columnDefs}
        defaultColDef={defaultColDef}
        rowSelection="single"
        getRowId={(params) => params.data.job_id}
        onRowClicked={(event) => {
          if (event.data && onJobSelect) {
            onJobSelect(event.data.job_id);
          }
        }}
        animateRows={true}
        pagination={true}
        paginationPageSize={25}
      />
    </div>
  );
}
