import {
  CButton,
  CCol,
  CListGroup,
  CListGroupItem,
  CNavItem,
  CRow,
  CSidebar,
  CSidebarBrand,
  CSidebarHeader,
  CSidebarNav,
  CSidebarToggler,
} from "@coreui/react";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import "./App.css";
import Dashboard from "./dashboard/Dashboard";

const router = createBrowserRouter([
  {
    path: "/config",
    element: <h1>Configurations coming soon...</h1>,
  },
  {
    path: "/",
    element: <Dashboard />,
  },
]);

function App() {
  return (
    <CRow className="body">
      <CSidebar className="border-end">
        <CSidebarHeader className="border-buttom">
          <CSidebarBrand>MDE for FL</CSidebarBrand>
        </CSidebarHeader>
        <CSidebarNav>
          <CNavItem href="/config">Configurations</CNavItem>
          <CNavItem href="/">Insights</CNavItem>
        </CSidebarNav>
      </CSidebar>
      <CCol className="router-provider">
        <RouterProvider router={router} />
      </CCol>
    </CRow>
  );
}

export default App;
