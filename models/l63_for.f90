!
! To compile:
! f2py -c l63_for.f90 -m l63_for --fcompiler=intelem (for intel fortran compiler: --fcompiler=intelem)
  subroutine tinteg_l63(nx,npar,xold,x,par,kt,dt) 

! single l63 integration
    
    integer, intent(in) :: nx,npar
!f2py intent(hide) nx, npar
    real*8,  intent(in), dimension(nx)  :: xold
    real*8, intent(in), dimension(npar)   :: par
    real*8, intent(out), dimension(nx)  :: x
    integer, intent(in) :: kt
    real*8, intent(in)  :: dt
    real*8, dimension(nx)  :: xf1,xf2,xf3,xf4,xtmp
    integer :: k
    
    x=xold
    do k=1,kt

       xtmp(:) = x(:)
       call l63_core(nx,xtmp,xf1,par,dt)
       xtmp(:) = x(:) +  0.5d0 * xf1(:)
       call l63_core(nx,xtmp,xf2,par,dt)
       xtmp(:) = x(:) +  0.5d0 * xf2(:)
       call l63_core(nx,xtmp,xf3,par,dt)
       xtmp(:) = x(:) +  xf3(:)
       call l63_core(nx,xtmp,xf4,par,dt)

       x(:) = x(:) + ( xf1(:) + 2.0d0 * xf2(:) + 2.0d0 * xf3(:) + xf4(:) ) / 6.0d0

    end do

  end subroutine tinteg_l63
!
!----------------------------------------------------
!
  subroutine l63_core(nx,xin,xout,par,dt)

    integer,intent(in) :: nx
    real*8,intent(in), dimension(nx) :: xin,par
    real*8,intent(in) :: dt
    real*8,intent(out), dimension(nx) :: xout
    
    xout(1)=par(1)*(xin(2)-xin(1))
    xout(2)=par(2)*xin(1)-xin(2)-xin(1)*xin(3)
    xout(3)=xin(1)*xin(2)-par(3)*xin(3)

    xout(:) = dt * xout(:)

  end subroutine l63_core
!
!----------------------------------------------------
!
  subroutine tintegem_l63(nx,npar,nem,xold,x,par,kt,dt) 
    
! ensemble l63 integration
! could be parallelized

    integer, intent(in) :: nx,npar,nem
!f2py intent(hide) nx,npar,nem
    real*8, intent(in),dimension(nx,nem) :: xold
    real*8, intent(out),dimension(nx,nem) :: x
    real*8, intent(in),dimension(npar)    :: par

    integer, intent(in) :: kt
    real*8, intent(in)    :: dt

    real*8, dimension(nx,nem) :: xtmp,xf1,xf2,xf3,xf4
    integer :: k

    x=xold

    do k=1,kt

       xtmp(:,:) = x(:,:)
       call l63_corem(nx,nem,xtmp,xf1,par,dt)
       xtmp(:,:) = x(:,:) + xf1(:,:)
       call l63_corem(nx,nem,xtmp,xf2,par,dt)
       xtmp(:,:) = x(:,:) + xf2(:,:)
       call l63_corem(nx,nem,xtmp,xf3,par,dt)
       xtmp(:,:) = x(:,:) + xf3(:,:)
       call l63_corem(nx,nem,xtmp,xf4,par,dt)

       x(:,:) = x(:,:) + ( xf1(:,:) + 2.0d0 * xf2(:,:) + 2.0d0 * xf3(:,:) + xf4(:,:) ) / 6.0d0

    end do

  end subroutine tintegem_l63
!
!----------------------------------------------------
!
  subroutine l63_corem(nx,nem,xin,xout,par,dt)

    integer,intent(in) :: nx,nem
    real*8,intent(in),dimension (nx,nem) :: xin
    real*8,intent(in),dimension (nx) ::    par
    real*8,intent(in) :: dt
    real*8,intent(out),dimension (nx,nem) :: xout
    
    xout(1,:)=par(1)*(xin(2,:)-xin(1,:))
    xout(2,:)=par(2)*xin(1,:)-xin(2,:)-xin(1,:)*xin(3,:)
    xout(3,:)=xin(1,:)*xin(2,:)-par(3)*xin(3,:)

    xout(:,:) = dt * xout(:,:)

  end subroutine l63_corem
